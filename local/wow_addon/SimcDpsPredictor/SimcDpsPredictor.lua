local ADDON_NAME, NS = ...
local Model = _G.SimcDpsModelData

if type(Model) ~= "table" then
  print("Mr. Mythical: DPS Predictor: model data missing. Run export_wow_addon.py first.")
  return
end

local Predictor = {}
NS.Predictor = Predictor

local didWarnTooltipError = false
local didWarnComparisonError = false
local didWarnComparisonParse = false
local didWarnComparisonMissingGuid = false
local didWarnComparisonParseSample = false
local didWarnComparisonEmpty = false

SIMCDPS_CONFIG = SIMCDPS_CONFIG or {
  spec_key = nil,           -- nil = auto-detect from current spec; set manually with /simcdps spec <key>
  show_tooltip = true,
  use_ensemble = true,
  bag_scan_yield_every = 40,
  bag_item_selection = {},  -- key -> false means excluded; nil/true means included
}

if type(SIMCDPS_CONFIG.bag_item_selection) ~= "table" then
  SIMCDPS_CONFIG.bag_item_selection = {}
end

-- Module-level derived state (not persisted; refreshed on login / spec change).
local active_spec_keys   = {}   -- all model profiles matching the player's current class+spec
local active_spec_prefix = nil  -- e.g. "MID1_Hunter_Survival"
local profileDetectionDoneRef = { false } -- shared flag; set true after first tooltip-driven detection

-- Maps WoW class file token → MID1 key fragment.
local CLASS_TOKEN_TO_KEY = {
  DEATHKNIGHT = "Death_Knight",
  DEMONHUNTER = "Demon_Hunter",
  DRUID       = "Druid",
  EVOKER      = "Evoker",
  HUNTER      = "Hunter",
  MAGE        = "Mage",
  MONK        = "Monk",
  PALADIN     = "Paladin",
  PRIEST      = "Priest",
  ROGUE       = "Rogue",
  SHAMAN      = "Shaman",
  WARLOCK     = "Warlock",
  WARRIOR     = "Warrior",
}

-- Maps WoW class → primary armor subClassID (the type they should actually wear).
-- subClassID: 1=Cloth, 2=Leather, 3=Mail, 4=Plate.
local CLASS_PRIMARY_ARMOR = {
  WARRIOR     = 4, PALADIN     = 4, DEATHKNIGHT = 4,
  HUNTER      = 3, SHAMAN      = 3, EVOKER      = 3,
  ROGUE       = 2, DRUID       = 2, MONK        = 2, DEMONHUNTER = 2,
  MAGE        = 1, WARLOCK     = 1, PRIEST      = 1,
}

local INVTYPE_TO_SLOT_IDS = {
  INVTYPE_HEAD = {1},
  INVTYPE_NECK = {2},
  INVTYPE_SHOULDER = {3},
  INVTYPE_CLOAK = {15},
  INVTYPE_CHEST = {5},
  INVTYPE_ROBE = {5},
  INVTYPE_WRIST = {9},
  INVTYPE_HAND = {10},
  INVTYPE_WAIST = {6},
  INVTYPE_LEGS = {7},
  INVTYPE_FEET = {8},
  INVTYPE_FINGER = {11, 12},
  INVTYPE_TRINKET = {13, 14},
  INVTYPE_WEAPON = {16, 17},
  INVTYPE_2HWEAPON = {16},
  INVTYPE_WEAPONMAINHAND = {16},
  INVTYPE_WEAPONOFFHAND = {17},
  INVTYPE_HOLDABLE = {17},
  INVTYPE_SHIELD = {17},
  INVTYPE_RANGED = {16},
  INVTYPE_RANGEDRIGHT = {16},
}

-- Returns the MID1 prefix for the player's current class+spec, e.g. "MID1_Hunter_Survival".
-- Returns nil when class/spec cannot be determined.
local function buildSpecPrefix(classToken, specName)
  local classKey = CLASS_TOKEN_TO_KEY[classToken]
  if not classKey then return nil end
  return "MID1_" .. classKey .. "_" .. specName:gsub(" ", "_")
end

-- Collect all spec keys in the model that match prefix (exact or as prefix + "_...").
local function findSpecProfiles(prefix)
  local matches = {}
  for _, sfName in ipairs(Model.spec_feature_names) do
    local specKey = sfName:gsub("^spec_", "")
    if specKey == prefix or specKey:sub(1, #prefix + 1) == prefix .. "_" then
      table.insert(matches, specKey)
    end
  end
  return matches
end

-- Build a short human-readable label for a profile, e.g. "PL DW", "Sentinel 2H", "Frostfire", "Frost".
local function getProfileLabel(specKey, prefix)
  if specKey == prefix then
    -- No variant suffix – show just the spec name part (last segment of prefix).
    return (prefix:match("[^_]+$") or specKey):gsub("_", " ")
  end
  -- Has variant suffix – strip "prefix_" and humanise underscores.
  return specKey:sub(#prefix + 2):gsub("_", " ")
end

-- Populate active_spec_keys / active_spec_prefix from the player's current specialisation.
local function detectAndCacheProfiles()
  profileDetectionDoneRef[1] = true
  local _, classToken = UnitClass("player")
  local specIndex = GetSpecialization()
  if not classToken or not specIndex then
    active_spec_keys   = {}
    active_spec_prefix = nil
    return
  end
  local _, specName = GetSpecializationInfo(specIndex)
  if not specName then
    active_spec_keys   = {}
    active_spec_prefix = nil
    return
  end
  local prefix = buildSpecPrefix(classToken, specName)
  if not prefix then
    active_spec_keys   = {}
    active_spec_prefix = nil
    return
  end
  active_spec_prefix = prefix
  active_spec_keys   = findSpecProfiles(prefix)
end

local function relu(x)
  if x > 0 then
    return x
  end
  return 0
end

local function dot(row, vec, bias)
  local s = bias
  for i = 1, #row do
    s = s + row[i] * vec[i]
  end
  return s
end

-- Pre-allocated ping-pong buffers to avoid table creation per layer.
local buf_a = {}
local buf_b = {}
for i = 1, 1024 do buf_a[i] = 0; buf_b[i] = 0 end

local function linear_into(layer_w, layer_b, input, out, out_size)
  for i = 1, out_size do
    out[i] = dot(layer_w[i], input, layer_b[i])
  end
end

local function batchnorm_precomputed(x, bn_s, bn_o, size)
  -- Precomputed: y = scale * x + offset (no sqrt at runtime)
  for i = 1, size do
    x[i] = bn_s[i] * x[i] + bn_o[i]
  end
end

local function batchnorm_raw(x, bn_w, bn_b, bn_rm, bn_rv, eps)
  -- v3 fallback: full BN with sqrt at runtime
  local out = {}
  for i = 1, #x do
    local norm = (x[i] - bn_rm[i]) / math.sqrt(bn_rv[i] + eps)
    out[i] = bn_w[i] * norm + bn_b[i]
  end
  return out
end

local function forwardModel(input, specKey, modelDef)
  local scaler = Model.scaler
  local is_v5 = (Model.model_version == "v5")
  local n_in = is_v5 and (Model.n_stat_features or #input) or Model.input_size

  -- Scale input into buf_a
  for i = 1, n_in do
    buf_a[i] = (input[i] - scaler.x_mean[i]) / scaler.x_scale[i]
  end

  local src = buf_a
  local dst = buf_b

  for li = 1, #modelDef.layers do
    local layer = modelDef.layers[li]
    local out_size = #layer.w

    -- Layer 1 with prebaked spec bias (v5)
    if li == 1 and is_v5 and modelDef.prebaked then
      local spec_bias = modelDef.prebaked[specKey]
      if spec_bias then
        for i = 1, out_size do
          dst[i] = dot(layer.w[i], src, spec_bias[i])
        end
      else
        linear_into(layer.w, layer.b, src, dst, out_size)
      end
    else
      linear_into(layer.w, layer.b, src, dst, out_size)
    end

    -- ReLU in-place
    for j = 1, out_size do
      if dst[j] <= 0 then dst[j] = 0 end
    end

    if is_v5 or (Model.model_version == "v4") then
      batchnorm_precomputed(dst, layer.bn_s, layer.bn_o, out_size)
    else
      local bn_out = batchnorm_raw(dst, layer.bn_w, layer.bn_b, layer.bn_rm, layer.bn_rv, Model.bn_eps or 1e-5)
      for j = 1, out_size do dst[j] = bn_out[j] end
    end

    -- Swap buffers
    src, dst = dst, src
  end

  -- src now points to the final hidden output
  local y_scaled = dot(modelDef.output.w, src, modelDef.output.b)
  local y = y_scaled * scaler.y_scale + scaler.y_mean
  return y
end

local function forward(input, specKey)
  if Model.single_model then
    return forwardModel(input, specKey, Model.single_model)
  end
  -- Compatibility path for old exports.
  return forwardModel(input, specKey, { layers = Model.layers, output = Model.output })
end

local function getPrimaryStatValue()
  local specIndex = GetSpecialization()
  if not specIndex then
    return 0
  end

  local _, _, _, _, _, _, primaryStat = GetSpecializationInfo(specIndex)
  if primaryStat == LE_UNIT_STAT_STRENGTH then
    local v = UnitStat("player", LE_UNIT_STAT_STRENGTH)
    return v or 0
  elseif primaryStat == LE_UNIT_STAT_AGILITY then
    local v = UnitStat("player", LE_UNIT_STAT_AGILITY)
    return v or 0
  end

  local v = UnitStat("player", LE_UNIT_STAT_INTELLECT)
  return v or 0
end

local function getPlayerStatVector()
  return {
    primary_stat = getPrimaryStatValue(),
    crit = GetCombatRating(CR_CRIT_MELEE) or 0,
    haste = GetCombatRating(CR_HASTE_MELEE) or 0,
    mastery = GetCombatRating(CR_MASTERY) or 0,
    versatility = GetCombatRating(CR_VERSATILITY_DAMAGE_DONE) or 0,
  }
end

local function itemRefToLink(itemRef)
  if type(itemRef) == "table" then
    return itemRef.link or itemRef.itemLink or itemRef.hyperlink
  end
  return itemRef
end

local SLOT_ID_TO_NAME = {
  [1] = "HeadSlot", [2] = "NeckSlot", [3] = "ShoulderSlot", [5] = "ChestSlot",
  [6] = "WaistSlot", [7] = "LegsSlot", [8] = "FeetSlot", [9] = "WristSlot",
  [10] = "HandsSlot", [11] = "Finger0Slot", [12] = "Finger1Slot", [13] = "Trinket0Slot",
  [14] = "Trinket1Slot", [15] = "BackSlot", [16] = "MainHandSlot", [17] = "SecondaryHandSlot",
}

local function getGuidFromEquipmentSlot(slotId)
  if C_Item and C_Item.GetItemGUID and ItemLocation and ItemLocation.CreateFromEquipmentSlot then
    local okLoc, loc = pcall(ItemLocation.CreateFromEquipmentSlot, slotId)
    if okLoc and loc then
      local okGuid, guid = pcall(C_Item.GetItemGUID, loc)
      if okGuid and guid then
        return guid
      end
    end
  end

  local getInventoryItemGUID = rawget(_G, "GetInventoryItemGUID")
  if getInventoryItemGUID then
    local invSlot = GetInventorySlotInfo(SLOT_ID_TO_NAME[slotId])
    if invSlot then
      local okGuid, guid = pcall(getInventoryItemGUID, "player", invSlot)
      if okGuid and guid then
        return guid
      end
    end
  end

  return nil
end

local function getGuidFromBagSlot(bag, slot)
  if C_Item and C_Item.GetItemGUID and ItemLocation and ItemLocation.CreateFromBagAndSlot then
    local okLoc, loc = pcall(ItemLocation.CreateFromBagAndSlot, bag, slot)
    if okLoc and loc then
      local okGuid, guid = pcall(C_Item.GetItemGUID, loc)
      if okGuid and guid then
        return guid
      end
    end
  end

  return nil
end

local function getNativeComparisonItemFromTooltipData(data)
  if type(data) == "table" and type(data.item) == "table" then
    return data.item
  end
  if type(data) == "table" and (data.guid or data.itemGUID or data.hyperlink or data.id or data.type) then
    return data
  end
  return nil
end

local function getNativeComparisonItemForInventorySlot(slotId)
  if not (C_TooltipInfo and C_TooltipInfo.GetInventoryItem) then
    return nil
  end
  local invSlot = GetInventorySlotInfo(SLOT_ID_TO_NAME[slotId])
  if not invSlot then
    return nil
  end
  local ok, data = pcall(C_TooltipInfo.GetInventoryItem, "player", invSlot)
  if not ok then
    return nil
  end
  return getNativeComparisonItemFromTooltipData(data)
end

local function getNativeComparisonItemForBagSlot(bag, slot)
  if not (C_TooltipInfo and C_TooltipInfo.GetBagItem) then
    return nil
  end
  local ok, data = pcall(C_TooltipInfo.GetBagItem, bag, slot)
  if not ok then
    return nil
  end
  return getNativeComparisonItemFromTooltipData(data)
end

local function getItemRefFromInventoryByLink(itemLink)
  if not itemLink then
    return nil
  end
  for slotId, slotName in pairs(SLOT_ID_TO_NAME) do
    local invSlot = GetInventorySlotInfo(slotName)
    if invSlot then
      local link = GetInventoryItemLink("player", invSlot)
      if link == itemLink then
        local guid = getGuidFromEquipmentSlot(slotId)
        local comparisonItem = getNativeComparisonItemForInventorySlot(slotId)
        return { link = link, guid = guid, slotId = slotId, comparisonItem = comparisonItem }
      end
    end
  end
  return nil
end

local function getItemRefFromBagsByLink(itemLink)
  if not itemLink or not C_Container or not C_Container.GetContainerNumSlots or not C_Container.GetContainerItemInfo then
    return nil
  end
  for bag = 0, 4 do
    for slot = 1, C_Container.GetContainerNumSlots(bag) do
      local info = C_Container.GetContainerItemInfo(bag, slot)
      if info and info.hyperlink == itemLink then
        local guid = info["itemGUID"] or info["guid"] or getGuidFromBagSlot(bag, slot)
        local comparisonItem = getNativeComparisonItemForBagSlot(bag, slot)
        return { link = itemLink, guid = guid, bag = bag, slot = slot, comparisonItem = comparisonItem }
      end
    end
  end
  return nil
end

local function resolveOwnedItemRef(itemRef)
  if type(itemRef) == "table" then
    local link = itemRef.link or itemRef.itemLink or itemRef.hyperlink
    local guid = itemRef.guid or itemRef.itemGUID
    local comparisonItem = itemRef.comparisonItem
    if guid then
      return {
        link = link,
        guid = guid,
        itemGUID = itemRef.itemGUID,
        comparisonItem = comparisonItem,
        slotId = itemRef.slotId,
        bag = itemRef.bag,
        slot = itemRef.slot,
      }
    end
    if link then
      local bagRef = getItemRefFromBagsByLink(link)
      if bagRef then
        bagRef.comparisonItem = comparisonItem
        return bagRef
      end
      local invRef = getItemRefFromInventoryByLink(link)
      if invRef then
        invRef.comparisonItem = comparisonItem
        return invRef
      end
      return { link = link, comparisonItem = comparisonItem }
    end
    return itemRef
  end

  local link = itemRef
  if not link then
    return nil
  end
  local bagRef = getItemRefFromBagsByLink(link)
  if bagRef then return bagRef end
  local invRef = getItemRefFromInventoryByLink(link)
  if invRef then return invRef end
  return { link = link }
end

-- Extract stat deltas using numeric item stat APIs.
-- Optional pairedItemLink supports explicit dual-slot math (e.g. 2H vs MH+OH).
local function getItemStatDeltas(itemLink, equippedItemLink, pairedItemLink, addPairedStats)
  if not (C_Item and C_Item.GetItemStatDelta and C_Item.GetItemStats) then
    return nil, "item stat API unavailable"
  end

  local comparisonLink = itemRefToLink(itemLink)
  local equippedLink = itemRefToLink(equippedItemLink)
  local pairedLink = itemRefToLink(pairedItemLink)

  if not comparisonLink or not equippedLink then
    return nil, "missing item link context"
  end

  local stats = {
    primary_stat = 0,
    crit = 0,
    haste = 0,
    mastery = 0,
    versatility = 0,
  }

  local KEY_TO_FEATURE = {
    ITEM_MOD_STRENGTH_SHORT = "primary_stat",
    ITEM_MOD_AGILITY_SHORT = "primary_stat",
    ITEM_MOD_INTELLECT_SHORT = "primary_stat",
    ITEM_MOD_CRIT_RATING_SHORT = "crit",
    ITEM_MOD_CR_CRIT_SHORT = "crit",
    ITEM_MOD_HASTE_RATING_SHORT = "haste",
    ITEM_MOD_MASTERY_RATING_SHORT = "mastery",
    ITEM_MOD_VERSATILITY = "versatility",
    ITEM_MOD_VERSATILITY_SHORT = "versatility",
  }

  local function applyStatTable(deltaTable, sign)
    if type(deltaTable) ~= "table" then
      return
    end
    for k, v in pairs(deltaTable) do
      if type(k) == "string" and type(v) == "number" then
        local feature = KEY_TO_FEATURE[k]
        if feature then
          stats[feature] = (stats[feature] or 0) + sign * v
        end
      end
    end
  end

  -- Primary comparison is always explicit: comparison item minus equipped item.
  local okDelta, statDelta = pcall(C_Item.GetItemStatDelta, comparisonLink, equippedLink)
  if not okDelta or type(statDelta) ~= "table" then
    if not didWarnComparisonError then
      didWarnComparisonError = true
      print("Mr. Mythical: DPS Predictor: GetItemStatDelta failed for compared/equipped links")
    end
    return nil, "failed to compute item stat delta"
  end
  applyStatTable(statDelta, 1)

  -- Optional paired item adjustment is explicit numeric math.
  -- addPairedStats=true: include paired stats; false/nil: subtract paired stats.
  if pairedLink then
    local okPaired, pairedStats = pcall(C_Item.GetItemStats, pairedLink)
    if not okPaired or type(pairedStats) ~= "table" then
      return nil, "failed to read paired item stats"
    end
    local pairedSign = (addPairedStats == true) and 1 or -1
    applyStatTable(pairedStats, pairedSign)
  end

  return stats, nil
end

local function addStats(a, b, sign)
  return {
    primary_stat = (a.primary_stat or 0) + sign * (b.primary_stat or 0),
    crit = (a.crit or 0) + sign * (b.crit or 0),
    haste = (a.haste or 0) + sign * (b.haste or 0),
    mastery = (a.mastery or 0) + sign * (b.mastery or 0),
    versatility = (a.versatility or 0) + sign * (b.versatility or 0),
  }
end

local function buildFeatureVector(stats, specKey)
  local features = {}
  for i = 1, Model.input_size do
    features[i] = 0
  end

  features[Model.feature_index["primary_stat"]] = stats.primary_stat or 0
  features[Model.feature_index["crit"]] = stats.crit or 0
  features[Model.feature_index["haste"]] = stats.haste or 0
  features[Model.feature_index["mastery"]] = stats.mastery or 0
  features[Model.feature_index["versatility"]] = stats.versatility or 0

  local specFeature = "spec_" .. specKey
  local specIdx = Model.feature_index[specFeature]
  if specIdx then
    features[specIdx] = 1
  end

  return features
end

local function predictWithStats(stats, specKey)
  local is_v5 = (Model.model_version == "v5")
  local x
  if is_v5 then
    -- v5: only 5 stat features needed (spec is prebaked into layer-1 bias)
    x = {
      stats.primary_stat or 0,
      stats.crit or 0,
      stats.haste or 0,
      stats.mastery or 0,
      stats.versatility or 0,
    }
  else
    x = buildFeatureVector(stats, specKey)
  end

  local deployment = Model.deployment
  local ensemble = Model.ensemble_models
  if SIMCDPS_CONFIG.use_ensemble and deployment and deployment.mode == "ensemble" and ensemble and #ensemble > 0 then
    local strategy = deployment.strategy or "equal"
    local preds = {}
    for i = 1, #ensemble do
      preds[i] = forwardModel(x, specKey, ensemble[i])
    end

    if strategy == "spec_router" and deployment.spec_specialists then
      local targetTrial = deployment.spec_specialists[specKey]
      if targetTrial then
        for i = 1, #ensemble do
          if ensemble[i].trial_number == targetTrial then
            return preds[i]
          end
        end
      end
    end

    if strategy == "spec_inverse_mae" then
      local weightedSum = 0
      local wSum = 0
      for i = 1, #ensemble do
        local specMae = ensemble[i].per_spec_mae and ensemble[i].per_spec_mae[specKey]
        local safeMae = specMae or ensemble[i].test_mae or 1e9
        if safeMae < 1e-8 then
          safeMae = 1e-8
        end
        local w = 1 / safeMae
        weightedSum = weightedSum + w * preds[i]
        wSum = wSum + w
      end
      if wSum > 0 then
        return weightedSum / wSum
      end
    end

    if strategy == "global_inverse_mae" then
      local weightedSum = 0
      local wSum = 0
      for i = 1, #ensemble do
        local safeMae = ensemble[i].test_mae or 1e9
        if safeMae < 1e-8 then
          safeMae = 1e-8
        end
        local w = 1 / safeMae
        weightedSum = weightedSum + w * preds[i]
        wSum = wSum + w
      end
      if wSum > 0 then
        return weightedSum / wSum
      end
    end

    -- Equal average fallback.
    local s = 0
    for i = 1, #preds do
      s = s + preds[i]
    end
    return s / #preds
  end

  return forward(x, specKey)
end

-- Base DPS cache: avoids recomputing base prediction when stats haven't changed.
-- Invalidated on gear change (PLAYER_EQUIPMENT_CHANGED) and spec change.
-- LRU cache with max size 256 to prevent unbounded growth.
local baseDpsCache = {}  -- key = statsHash..":"..specKey, value = {dps=number, order=N}
local baseDpsCacheDirty = true
local baseDpsCacheOrder = 0
local MAX_BASE_DPS_CACHE_SIZE = 256

local function statsHash(stats)
  return (stats.primary_stat or 0) + (stats.crit or 0) * 0.001
       + (stats.haste or 0) * 0.0001 + (stats.mastery or 0) * 0.00001
       + (stats.versatility or 0) * 0.000001
end

local function evictOldestBaseDps()
  local count = 0
  for _ in pairs(baseDpsCache) do count = count + 1 end
  if count <= MAX_BASE_DPS_CACHE_SIZE then return end
  
  local oldest_key, oldest_order = nil, math.huge
  for k, v in pairs(baseDpsCache) do
    if v.order < oldest_order then
      oldest_key = k
      oldest_order = v.order
    end
  end
  if oldest_key then
    baseDpsCache[oldest_key] = nil
  end
end

local function getCachedBaseDps(stats, specKey)
  if baseDpsCacheDirty then
    baseDpsCache = {}
    baseDpsCacheDirty = false
    baseDpsCacheOrder = 0
  end
  local key = statsHash(stats) .. ":" .. specKey
  local cached = baseDpsCache[key]
  if cached then
    cached.order = baseDpsCacheOrder
    baseDpsCacheOrder = baseDpsCacheOrder + 1
    return cached.dps
  end
  local dps = predictWithStats(stats, specKey)
  evictOldestBaseDps()
  baseDpsCache[key] = { dps = dps, order = baseDpsCacheOrder }
  baseDpsCacheOrder = baseDpsCacheOrder + 1
  return dps
end

local function getEquipSlotCandidates(itemRef)
  local itemLink = itemRefToLink(itemRef)
  local _, _, _, _, _, _, _, _, equipLoc = GetItemInfo(itemLink)
  if not equipLoc then
    return nil
  end
  return INVTYPE_TO_SLOT_IDS[equipLoc]
end

local function getSlotItemRef(slotId)
  local invSlot = GetInventorySlotInfo(SLOT_ID_TO_NAME[slotId])
  if not invSlot then
    return nil
  end
  local link = GetInventoryItemLink("player", invSlot)
  if not link then
    return nil
  end
  local guid = getGuidFromEquipmentSlot(slotId)
  local comparisonItem = getNativeComparisonItemForInventorySlot(slotId)
  return { link = link, guid = guid, slotId = slotId, comparisonItem = comparisonItem }
end

local function is2HWeapon(itemLink)
  if not itemLink then return false end
  local _, _, _, _, _, _, _, _, equipLoc = GetItemInfo(itemLink)
  return equipLoc == "INVTYPE_2HWEAPON"
end

local WEAPON_SUBCLASS = {
  AXE1 = 0,
  AXE2 = 1,
  MACE1 = 4,
  MACE2 = 5,
  POLEARM = 6,
  SWORD1 = 7,
  SWORD2 = 8,
  STAFF = 10,
  FIST = 13,
  DAGGER = 15,
  BOW = 2,
  GUN = 3,
  CROSSBOW = 18,
  WAND = 19,
  WARGLAIVE = 9,
}

local CLASS_WEAPON_RULES = {
  DEATHKNIGHT = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true, [WEAPON_SUBCLASS.AXE2] = true,
      [WEAPON_SUBCLASS.MACE1] = true, [WEAPON_SUBCLASS.MACE2] = true,
      [WEAPON_SUBCLASS.SWORD1] = true, [WEAPON_SUBCLASS.SWORD2] = true,
      [WEAPON_SUBCLASS.POLEARM] = true,
    },
    allow_shield = false,
    allow_holdable = false,
  },
  DEMONHUNTER = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true,
      [WEAPON_SUBCLASS.SWORD1] = true,
      [WEAPON_SUBCLASS.FIST] = true,
      [WEAPON_SUBCLASS.WARGLAIVE] = true,
    },
    allow_shield = false,
    allow_holdable = false,
  },
  DRUID = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.FIST] = true,
      [WEAPON_SUBCLASS.MACE1] = true,
      [WEAPON_SUBCLASS.MACE2] = true,
      [WEAPON_SUBCLASS.POLEARM] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
    },
    allow_shield = false,
    allow_holdable = false,
  },
  EVOKER = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true,
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.FIST] = true,
      [WEAPON_SUBCLASS.MACE1] = true,
      [WEAPON_SUBCLASS.SWORD1] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
    },
    allow_shield = false,
    allow_holdable = true,
  },
  HUNTER = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true, [WEAPON_SUBCLASS.AXE2] = true,
      [WEAPON_SUBCLASS.SWORD1] = true, [WEAPON_SUBCLASS.SWORD2] = true,
      [WEAPON_SUBCLASS.POLEARM] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.BOW] = true,
      [WEAPON_SUBCLASS.GUN] = true,
      [WEAPON_SUBCLASS.CROSSBOW] = true,
    },
    allow_shield = false,
    allow_holdable = false,
  },
  MAGE = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.SWORD1] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.WAND] = true,
    },
    allow_shield = false,
    allow_holdable = true,
  },
  MONK = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true,
      [WEAPON_SUBCLASS.MACE1] = true,
      [WEAPON_SUBCLASS.SWORD1] = true,
      [WEAPON_SUBCLASS.POLEARM] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.FIST] = true,
    },
    allow_shield = false,
    allow_holdable = false,
  },
  PALADIN = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true, [WEAPON_SUBCLASS.AXE2] = true,
      [WEAPON_SUBCLASS.MACE1] = true, [WEAPON_SUBCLASS.MACE2] = true,
      [WEAPON_SUBCLASS.SWORD1] = true, [WEAPON_SUBCLASS.SWORD2] = true,
      [WEAPON_SUBCLASS.POLEARM] = true,
    },
    allow_shield = true,
    allow_holdable = false,
  },
  PRIEST = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.MACE1] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.WAND] = true,
    },
    allow_shield = false,
    allow_holdable = true,
  },
  ROGUE = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true,
      [WEAPON_SUBCLASS.MACE1] = true,
      [WEAPON_SUBCLASS.SWORD1] = true,
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.FIST] = true,
    },
    allow_shield = false,
    allow_holdable = false,
  },
  SHAMAN = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true, [WEAPON_SUBCLASS.AXE2] = true,
      [WEAPON_SUBCLASS.MACE1] = true, [WEAPON_SUBCLASS.MACE2] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.FIST] = true,
    },
    allow_shield = true,
    allow_holdable = true,
  },
  WARLOCK = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.SWORD1] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.WAND] = true,
    },
    allow_shield = false,
    allow_holdable = true,
  },
  WARRIOR = {
    weapon_subclasses = {
      [WEAPON_SUBCLASS.AXE1] = true, [WEAPON_SUBCLASS.AXE2] = true,
      [WEAPON_SUBCLASS.MACE1] = true, [WEAPON_SUBCLASS.MACE2] = true,
      [WEAPON_SUBCLASS.SWORD1] = true, [WEAPON_SUBCLASS.SWORD2] = true,
      [WEAPON_SUBCLASS.POLEARM] = true,
      [WEAPON_SUBCLASS.STAFF] = true,
      [WEAPON_SUBCLASS.DAGGER] = true,
      [WEAPON_SUBCLASS.FIST] = true,
    },
    allow_shield = true,
    allow_holdable = false,
  },
}

local SPEC_OFFHAND_WEAPON_ALLOWED = {
  Death_Knight_Frost = true,
  Demon_Hunter_Havoc = true,
  Demon_Hunter_Vengeance = true,
  Monk_Brewmaster = true,
  Monk_Windwalker = true,
  Rogue_Assassination = true,
  Rogue_Outlaw = true,
  Rogue_Subtlety = true,
  Shaman_Enhancement = true,
  Warrior_Fury = true,
}

local function getClassSpecPair(specKey)
  if type(specKey) ~= "string" then
    return nil, nil
  end
  local short = specKey:gsub("^MID1_", "")
  local parts = {}
  for token in short:gmatch("[^_]+") do
    table.insert(parts, token)
  end
  if #parts < 2 then
    return nil, nil
  end

  local classPart = parts[1]
  local specPart = parts[2]
  if classPart == "Death" and parts[2] == "Knight" and parts[3] then
    classPart = "Death_Knight"
    specPart = parts[3]
  elseif classPart == "Demon" and parts[2] == "Hunter" and parts[3] then
    classPart = "Demon_Hunter"
    specPart = parts[3]
  end
  return classPart, specPart
end

local function getItemTypeInfo(itemLink)
  if not itemLink then
    return nil, nil, nil
  end

  local _, _, _, itemEquipLoc, _, itemClassID, itemSubClassID = GetItemInfoInstant(itemLink)
  if itemEquipLoc and itemEquipLoc ~= "" then
    return itemClassID, itemSubClassID, itemEquipLoc
  end

  local _, _, _, _, _, itemClassID2, itemSubClassID2, _, equipLoc2 = GetItemInfo(itemLink)
  return itemClassID2, itemSubClassID2, equipLoc2
end

local function isWeaponSubclassAllowedForClass(classToken, subClassID)
  local rules = CLASS_WEAPON_RULES[classToken]
  if not rules or type(rules.weapon_subclasses) ~= "table" then
    return false
  end
  return rules.weapon_subclasses[subClassID] == true
end

local function isSpecAllowedWeaponOffhand(specKey)
  local classPart, specPart = getClassSpecPair(specKey)
  if not classPart or not specPart then
    return false
  end
  return SPEC_OFFHAND_WEAPON_ALLOWED[classPart .. "_" .. specPart] == true
end

local function isOffhandTypeAllowedForClass(classToken, equipLoc)
  local rules = CLASS_WEAPON_RULES[classToken]
  if not rules then
    return false
  end
  if equipLoc == "INVTYPE_SHIELD" then
    return rules.allow_shield == true
  end
  if equipLoc == "INVTYPE_HOLDABLE" then
    return rules.allow_holdable == true
  end
  return false
end

local function isItemAllowedForMainHand(classToken, itemClassID, itemSubClassID, equipLoc)
  if not classToken then
    return false
  end
  if equipLoc == "INVTYPE_SHIELD" or equipLoc == "INVTYPE_HOLDABLE" then
    return false
  end
  if itemClassID ~= 2 then -- Weapon
    return false
  end
  return isWeaponSubclassAllowedForClass(classToken, itemSubClassID)
end

local function isItemAllowedForOffHand(classToken, specKey, itemClassID, itemSubClassID, equipLoc)
  if not classToken then
    return false
  end
  if equipLoc == "INVTYPE_SHIELD" or equipLoc == "INVTYPE_HOLDABLE" then
    return isOffhandTypeAllowedForClass(classToken, equipLoc)
  end
  if itemClassID ~= 2 then -- Weapon
    return false
  end
  if not isWeaponSubclassAllowedForClass(classToken, itemSubClassID) then
    return false
  end
  return isSpecAllowedWeaponOffhand(specKey)
end

local function isSameItemRef(a, b)
  if not a or not b then
    return false
  end
  local aGuid = a.guid or a.itemGUID
  local bGuid = b.guid or b.itemGUID
  if aGuid and bGuid then
    return aGuid == bGuid
  end
  return itemRefToLink(a) == itemRefToLink(b)
end

local function collectOwnedWeaponCandidates()
  local out = {}

  local mh = getSlotItemRef(16)
  if mh and mh.link then
    table.insert(out, mh)
  end
  local oh = getSlotItemRef(17)
  if oh and oh.link then
    table.insert(out, oh)
  end

  if C_Container and C_Container.GetContainerNumSlots and C_Container.GetContainerItemInfo then
    for bag = 0, 4 do
      for slot = 1, C_Container.GetContainerNumSlots(bag) do
        local info = C_Container.GetContainerItemInfo(bag, slot)
        if info and info.hyperlink then
          table.insert(out, {
            link = info.hyperlink,
            guid = info["itemGUID"] or info["guid"] or getGuidFromBagSlot(bag, slot),
            bag = bag,
            slot = slot,
          })
        end
      end
    end
  end

  return out
end

local function findBestPairScenario(candidateRef, mainHandRef, specKey, classToken, candidateIsOffHandOnly, base, basePred)
  local best = nil
  local lastErr = nil

  for _, pairRef in ipairs(collectOwnedWeaponCandidates()) do
    if not isSameItemRef(pairRef, candidateRef) then
      local pairLink = itemRefToLink(pairRef)
      if pairLink then
        local pairClassID, pairSubClassID, pairEquipLoc = getItemTypeInfo(pairLink)
        local validPair = false

        if candidateIsOffHandOnly then
          -- Need a 1H mainhand-capable weapon to pair with the hovered offhand item.
          validPair = (not is2HWeapon(pairLink))
            and isItemAllowedForMainHand(classToken, pairClassID, pairSubClassID, pairEquipLoc)
        else
          -- Need a valid offhand item to pair with the hovered 1H mainhand item.
          validPair = isItemAllowedForOffHand(classToken, specKey, pairClassID, pairSubClassID, pairEquipLoc)
        end

        if validPair then
          local stats, err
          if candidateIsOffHandOnly then
            -- (pair MH - equipped 2H) + hovered offhand
            stats, err = getItemStatDeltas(pairRef, mainHandRef, candidateRef, true)
          else
            -- (hovered 1H - equipped 2H) + pair offhand
            stats, err = getItemStatDeltas(candidateRef, mainHandRef, pairRef, true)
          end

          if stats then
            local newStats = addStats(base, stats, 1)
            local pred = predictWithStats(newStats, specKey)
            local delta = pred - basePred
            if not best or delta > best.dps_delta then
              best = {
                dps_base = basePred,
                dps_new = pred,
                dps_delta = delta,
                slot_id = 16,
                mode = "dw_pair_replacement",
              }
            end
          else
            lastErr = err or lastErr
          end
        end
      end
    end
  end

  return best, lastErr
end

-- Weapon loadout profiles:
-- two_handed = spec can use a 2H loadout
-- dual_wield = spec can use a 2x1H loadout (including off-hand/shield/holdable in slot 17)
local SPEC_WEAPON_LOADOUTS = {
  Death_Knight_Blood = { two_handed = true, dual_wield = false },
  Death_Knight_Frost = { two_handed = true, dual_wield = true },
  Death_Knight_Unholy = { two_handed = true, dual_wield = false },

  Demon_Hunter_Havoc = { two_handed = false, dual_wield = true },
  Demon_Hunter_Vengeance = { two_handed = false, dual_wield = true },

  Druid_Balance = { two_handed = true, dual_wield = true },
  Druid_Feral = { two_handed = true, dual_wield = false },
  Druid_Guardian = { two_handed = true, dual_wield = false },

  Evoker_Devastation = { two_handed = true, dual_wield = true },

  Hunter_Beast_Mastery = { two_handed = true, dual_wield = false },
  Hunter_Marksmanship = { two_handed = true, dual_wield = false },
  Hunter_Survival = { two_handed = true, dual_wield = false },

  Mage_Arcane = { two_handed = true, dual_wield = true },
  Mage_Fire = { two_handed = true, dual_wield = true },
  Mage_Frost = { two_handed = true, dual_wield = true },

  Monk_Brewmaster = { two_handed = true, dual_wield = true },
  Monk_Windwalker = { two_handed = true, dual_wield = true },

  Paladin_Protection = { two_handed = false, dual_wield = true },
  Paladin_Retribution = { two_handed = true, dual_wield = false },

  Priest_Shadow = { two_handed = true, dual_wield = true },

  Rogue_Assassination = { two_handed = false, dual_wield = true },
  Rogue_Outlaw = { two_handed = false, dual_wield = true },
  Rogue_Subtlety = { two_handed = false, dual_wield = true },

  Shaman_Elemental = { two_handed = true, dual_wield = true },
  Shaman_Enhancement = { two_handed = false, dual_wield = true },

  Warlock_Affliction = { two_handed = true, dual_wield = true },
  Warlock_Demonology = { two_handed = true, dual_wield = true },
  Warlock_Destruction = { two_handed = true, dual_wield = true },

  Warrior_Arms = { two_handed = true, dual_wield = false },
  Warrior_Fury = { two_handed = true, dual_wield = true },
  Warrior_Protection = { two_handed = false, dual_wield = true },
}

local function getWeaponLoadoutForSpec(specKey)
  if type(specKey) ~= "string" then
    return { two_handed = true, dual_wield = true }
  end

  local short = specKey:gsub("^MID1_", "")
  local parts = {}
  for token in short:gmatch("[^_]+") do
    table.insert(parts, token)
  end
  if #parts < 3 then
    return { two_handed = true, dual_wield = true }
  end

  local classPart = parts[1]
  local specPart = parts[2]
  if classPart == "Death" and parts[2] == "Knight" and parts[3] then
    classPart = "Death_Knight"
    specPart = parts[3]
  elseif classPart == "Demon" and parts[2] == "Hunter" and parts[3] then
    classPart = "Demon_Hunter"
    specPart = parts[3]
  end

  local key = classPart .. "_" .. specPart
  return SPEC_WEAPON_LOADOUTS[key] or { two_handed = true, dual_wield = true }
end

local function evaluateItem(itemRef, specKey)
  itemRef = resolveOwnedItemRef(itemRef)
  local itemLink = itemRefToLink(itemRef)
  local base = getPlayerStatVector()
  local basePred = getCachedBaseDps(base, specKey)
  local slots = getEquipSlotCandidates(itemRef)
  local _, _, candidateEquipLoc = getItemTypeInfo(itemLink)

  if not slots or #slots == 0 then
    return nil, "unknown equip slot"
  end

  local lastErr = nil

  -- For weapons/offhands: evaluate weapon scenarios.
  if candidateEquipLoc == "INVTYPE_WEAPON" or candidateEquipLoc == "INVTYPE_2HWEAPON"
    or candidateEquipLoc == "INVTYPE_WEAPONMAINHAND" or candidateEquipLoc == "INVTYPE_WEAPONOFFHAND"
    or candidateEquipLoc == "INVTYPE_HOLDABLE" or candidateEquipLoc == "INVTYPE_SHIELD" then
    local _, classToken = UnitClass("player")
    local itemClassID, itemSubClassID = getItemTypeInfo(itemLink)
    local mainHandRef = getSlotItemRef(16)
    local offHandRef = getSlotItemRef(17)
    local currentIs2H = is2HWeapon(mainHandRef and mainHandRef.link)
    local candidateIs2H = is2HWeapon(itemLink)
    local candidateIsOffHandOnly = (candidateEquipLoc == "INVTYPE_WEAPONOFFHAND" or candidateEquipLoc == "INVTYPE_HOLDABLE" or candidateEquipLoc == "INVTYPE_SHIELD")
    local loadout = getWeaponLoadoutForSpec(specKey)
    local results = {}

    if candidateIsOffHandOnly then
      if not isItemAllowedForOffHand(classToken, specKey, itemClassID, itemSubClassID, candidateEquipLoc) then
        return nil, "off-hand item type is not allowed for this class/spec"
      end
    else
      if not isItemAllowedForMainHand(classToken, itemClassID, itemSubClassID, candidateEquipLoc) then
        return nil, "weapon type is not allowed for this class"
      end
    end

    -- Candidate is 2H: compare against current loadout if spec supports 2H.
    if candidateIs2H then
      if not loadout.two_handed then
        return nil, "2H weapons are not supported for this spec"
      end

      if mainHandRef then
        local stats, err
        if offHandRef and not currentIs2H then
          stats, err = getItemStatDeltas(itemRef, mainHandRef, offHandRef, false)
        else
          stats, err = getItemStatDeltas(itemRef, mainHandRef)
        end

        if stats then
          local withNew = addStats(base, stats, 1)
          local pred = predictWithStats(withNew, specKey)
          local delta = pred - basePred
          table.insert(results, { dps_base = basePred, dps_new = pred, dps_delta = delta, slot_id = 16, mode = "2h_replacement" })
        else
          lastErr = err or lastErr
        end
      end

      if #results > 0 then
        return results
      end
      return nil, (lastErr or "native comparison unavailable for weapon")
    end

    -- Candidate is offhand-only: compare against offhand slot when dual loadout is supported.
    if candidateIsOffHandOnly then
      if not loadout.dual_wield then
        return nil, "off-hand weapons are not supported for this spec"
      end
      if currentIs2H then
        if mainHandRef then
          local pairResult, pairErr = findBestPairScenario(itemRef, mainHandRef, specKey, classToken, true, base, basePred)
          if pairResult then
            table.insert(results, pairResult)
            return results
          end
          return nil, (pairErr or "no compatible main-hand pair found for this off-hand item")
        end
        return nil, "native comparison unavailable for weapon"
      end
        if offHandRef then
        local ohStats, err = getItemStatDeltas(itemRef, offHandRef)
        if ohStats then
          local withNew = addStats(base, ohStats, 1)
          local pred = predictWithStats(withNew, specKey)
          local delta = pred - basePred
          table.insert(results, { dps_base = basePred, dps_new = pred, dps_delta = delta, slot_id = 17, mode = "oh_replacement" })
        else
          lastErr = err or lastErr
        end
        else
          local ohStats = {
            primary_stat = 0,
            crit = 0,
            haste = 0,
            mastery = 0,
            versatility = 0,
          }
          local okStats, statTable = pcall(C_Item.GetItemStats, itemLink)
          if okStats and type(statTable) == "table" then
            ohStats.primary_stat = (statTable.ITEM_MOD_STRENGTH_SHORT or 0) + (statTable.ITEM_MOD_AGILITY_SHORT or 0) + (statTable.ITEM_MOD_INTELLECT_SHORT or 0)
            ohStats.crit = (statTable.ITEM_MOD_CRIT_RATING_SHORT or 0) + (statTable.ITEM_MOD_CR_CRIT_SHORT or 0)
            ohStats.haste = statTable.ITEM_MOD_HASTE_RATING_SHORT or 0
            ohStats.mastery = statTable.ITEM_MOD_MASTERY_RATING_SHORT or 0
            ohStats.versatility = (statTable.ITEM_MOD_VERSATILITY or 0) + (statTable.ITEM_MOD_VERSATILITY_SHORT or 0)
          end
          local withNew = addStats(base, ohStats, 1)
          local pred = predictWithStats(withNew, specKey)
          local delta = pred - basePred
          table.insert(results, { dps_base = basePred, dps_new = pred, dps_delta = delta, slot_id = 17, mode = "oh_replacement" })
      end

      if #results > 0 then
        return results
      end
      return nil, (lastErr or "native comparison unavailable for weapon")
    end

    -- Candidate is 1H mainhand-compatible: evaluate dual-wield scenarios when supported.
    if loadout.dual_wield then
      -- Scenario 1: mainhand replacement (only when currently dual-wielding).
      if mainHandRef and not currentIs2H then
        local mhStats, err = getItemStatDeltas(itemRef, mainHandRef)
        if mhStats then
          local newStats = addStats(base, mhStats, 1)
          local pred1 = predictWithStats(newStats, specKey)
          local delta1 = pred1 - basePred
          table.insert(results, { dps_base = basePred, dps_new = pred1, dps_delta = delta1, slot_id = 16, mode = "mh_replacement" })
        else
          lastErr = err or lastErr
        end

        -- If the offhand slot is empty, also evaluate the best paired offhand.
        if not offHandRef then
          local pairResult, pairErr = findBestPairScenario(itemRef, mainHandRef, specKey, classToken, false, base, basePred)
          if pairResult then
            table.insert(results, pairResult)
          else
            lastErr = pairErr or lastErr
          end
        end
      end

      -- Scenario 1b: if currently using a 2H, also evaluate best dual-wield pair.
      if currentIs2H and mainHandRef then
        local pairResult, pairErr = findBestPairScenario(itemRef, mainHandRef, specKey, classToken, false, base, basePred)
        if pairResult then
          table.insert(results, pairResult)
        else
          lastErr = pairErr or lastErr
        end
      end

      -- Scenario 2: offhand replacement (when current loadout is dual).
      if offHandRef and not currentIs2H and isItemAllowedForOffHand(classToken, specKey, itemClassID, itemSubClassID, candidateEquipLoc) then
        local ohStats, err = getItemStatDeltas(itemRef, offHandRef)
        if ohStats then
          local newStats = addStats(base, ohStats, 1)
          local pred2 = predictWithStats(newStats, specKey)
          local delta2 = pred2 - basePred
          table.insert(results, { dps_base = basePred, dps_new = pred2, dps_delta = delta2, slot_id = 17, mode = "oh_replacement" })
        else
          lastErr = err or lastErr
        end
      end
    elseif loadout.two_handed then
      -- 2H-only specs can still compare 1H items directly against mainhand,
      -- but no dual/offhand scenarios are evaluated.
      if mainHandRef then
        local mhStats, err = getItemStatDeltas(itemRef, mainHandRef)
        if mhStats then
          local newStats = addStats(base, mhStats, 1)
          local pred1 = predictWithStats(newStats, specKey)
          local delta1 = pred1 - basePred
          table.insert(results, { dps_base = basePred, dps_new = pred1, dps_delta = delta1, slot_id = 16, mode = "mh_replacement" })
        else
          lastErr = err or lastErr
        end
      else
        return nil, "native comparison unavailable for weapon"
      end
    end

    if results and #results > 0 then
      return results
    end
    return nil, (lastErr or "native comparison unavailable for weapon")
  end

  -- For rings: evaluate both slots separately
  if slots[1] == 11 and slots[2] == 12 then
    local ring1Ref = getSlotItemRef(11)
    local ring2Ref = getSlotItemRef(12)
    local results = {}

    -- Ring slot 1
    if ring1Ref then
      local ring1Stats, err = getItemStatDeltas(itemRef, ring1Ref)
      if ring1Stats then
        local newStats = addStats(base, ring1Stats, 1)
        local pred1 = predictWithStats(newStats, specKey)
        local delta1 = pred1 - basePred
        table.insert(results, { dps_base = basePred, dps_new = pred1, dps_delta = delta1, slot_id = 11, mode = "ring1" })
      else
        lastErr = err or lastErr
      end
    end

    -- Ring slot 2
    if ring2Ref then
      local ring2Stats, err = getItemStatDeltas(itemRef, ring2Ref)
      if ring2Stats then
        local newStats = addStats(base, ring2Stats, 1)
        local pred2 = predictWithStats(newStats, specKey)
        local delta2 = pred2 - basePred
        table.insert(results, { dps_base = basePred, dps_new = pred2, dps_delta = delta2, slot_id = 12, mode = "ring2" })
      else
        lastErr = err or lastErr
      end
    end

    if results and #results > 0 then
      return results
    end
    return nil, (lastErr or "native comparison unavailable for rings")
  end

  -- For all other items: find the best slot match
  local best = nil
  for _, slotId in ipairs(slots) do
    local eqRef = getSlotItemRef(slotId)
    if eqRef then
      local itemStats, err = getItemStatDeltas(itemRef, eqRef)
      if itemStats then
        local newStats = addStats(base, itemStats, 1)
        local newPred = predictWithStats(newStats, specKey)
        local delta = newPred - basePred

        if not best or delta > best.dps_delta then
          best = {
            dps_base = basePred,
            dps_new = newPred,
            dps_delta = delta,
            slot_id = slotId,
            mode = "replacement",
          }
        end
      else
        lastErr = err or lastErr
      end
    end
  end

  if best then
    return best
  end
  return nil, (lastErr or "native comparison unavailable")
end

function Predictor.PredictItemDelta(itemRef, specKey)
  -- Resolve: explicit arg > manual pin > first auto-detected profile.
  specKey = specKey or SIMCDPS_CONFIG.spec_key or active_spec_keys[1]
  local itemLink = itemRefToLink(itemRef)
  if not itemLink then
    return nil, "missing item link"
  end
  if not specKey or specKey == "" then
    return nil, "missing spec key"
  end

  local pred, err = evaluateItem(itemRef, specKey)
  if not pred then
    return nil, err or "prediction failed"
  end
  return pred
end

local function formatDelta(v)
  if v >= 0 then
    return string.format("+%.0f", v)
  end
  return string.format("%.0f", v)
end

local function printPredictionForLink(itemLink)
  -- Collect which profiles to show.
  -- A manual pin (spec_key) restricts to that single profile; otherwise show all detected.
  local specKeys
  if SIMCDPS_CONFIG.spec_key then
    specKeys = { SIMCDPS_CONFIG.spec_key }
  elseif #active_spec_keys > 0 then
    specKeys = active_spec_keys
  else
    print("Mr. Mythical: DPS Predictor: no spec detected. Use /simcdps spec <key> or reload in-game.")
    return
  end

  for _, specKey in ipairs(specKeys) do
    local pred, err = Predictor.PredictItemDelta(itemLink, specKey)
    if not pred then
      print("Mr. Mythical: DPS Predictor: " .. tostring(err))
    else
      local label = (active_spec_prefix and getProfileLabel(specKey, active_spec_prefix)) or specKey
      if type(pred) == "table" and pred[1] then
        for _, p in ipairs(pred) do
          local modeStr = ""
          if p.mode == "mh_replacement" then
            modeStr = " (mainhand)"
          elseif p.mode == "2h_replacement" then
            modeStr = " (2H)"
          elseif p.mode == "dw_pair_replacement" then
            modeStr = " (paired 1H set)"
          elseif p.mode == "oh_replacement" then
            modeStr = " (offhand)"
          elseif p.mode == "ring1" then
            modeStr = " (ring 1)"
          elseif p.mode == "ring2" then
            modeStr = " (ring 2)"
          end
          print(string.format("Mr. Mythical: DPS Predictor [%s]%s: base=%.0f new=%.0f delta=%s",
            label, modeStr, p.dps_base, p.dps_new, formatDelta(p.dps_delta)))
        end
      else
        print(string.format("Mr. Mythical: DPS Predictor [%s]: base=%.0f new=%.0f delta=%s",
          label, pred.dps_base, pred.dps_new, formatDelta(pred.dps_delta)))
      end
    end
  end
end

local function handleSlash(msg)
  local cmd, rest = msg:match("^(%S*)%s*(.-)$")
  cmd = (cmd or ""):lower()

  if cmd == "spec" then
    if rest == "auto" or rest == "" then
      SIMCDPS_CONFIG.spec_key = nil
      detectAndCacheProfiles()
      local n = #active_spec_keys
      if n > 0 then
        print(string.format("Mr. Mythical: DPS Predictor: auto-detected %d profile(s) for %s", n, active_spec_prefix or "?"))
        for _, k in ipairs(active_spec_keys) do print("  " .. k) end
      else
        print("Mr. Mythical: DPS Predictor: no profiles found for current spec")
      end
    else
      SIMCDPS_CONFIG.spec_key = rest
      print("Mr. Mythical: DPS Predictor: spec pinned to " .. rest)
    end
    return
  end

  if cmd == "predict" and rest and rest ~= "" then
    printPredictionForLink(rest)
    return
  end

  if cmd == "list" then
    print("Mr. Mythical: DPS Predictor spec keys:")
    for _, s in ipairs(Model.spec_feature_names) do
      print("  " .. s:gsub("^spec_", ""))
    end
    return
  end

  if cmd == "profiles" then
    if #active_spec_keys == 0 then
      print("Mr. Mythical: DPS Predictor: no profiles detected for current spec. Use /simcdps spec auto to retry.")
    else
      print(string.format("Mr. Mythical: DPS Predictor: %d profile(s) for %s", #active_spec_keys, active_spec_prefix or "?"))
      for _, k in ipairs(active_spec_keys) do
        local label = active_spec_prefix and getProfileLabel(k, active_spec_prefix) or k
        local pinned = (SIMCDPS_CONFIG.spec_key == k) and " [pinned]" or ""
        print(string.format("  %s  (%s)%s", label, k, pinned))
      end
    end
    return
  end

  print("Mr. Mythical: DPS Predictor commands:")
  print("  /simcdps spec <key>    -- pin to a specific profile")
  print("  /simcdps spec auto     -- auto-detect from current spec (shows all profiles)")
  print("  /simcdps profiles      -- list profiles detected for your current spec")
  print("  /simcdps predict <itemLink>")
  print("  /simcdps list          -- all available spec keys")
  print("  /simcdps tooltip on|off|status")
  print("  /simcdps ensemble on|off")
  print("  /simcdps bags          -- open bag comparison UI")
  print("  /simcdps bagclose      -- close bag comparison UI")
  print("  /simcdps bagyield <n>|status")
end

local oldHandleSlash = handleSlash
handleSlash = function(msg)
  local cmd, rest = msg:match("^(%S*)%s*(.-)$")
  cmd = (cmd or ""):lower()

  if cmd == "ensemble" then
    local v = (rest or ""):lower()
    if v == "on" then
      SIMCDPS_CONFIG.use_ensemble = true
      print("Mr. Mythical: DPS Predictor: ensemble enabled")
      return
    end
    if v == "off" then
      SIMCDPS_CONFIG.use_ensemble = false
      print("Mr. Mythical: DPS Predictor: ensemble disabled (single model)")
      return
    end
    print("Mr. Mythical: DPS Predictor: /simcdps ensemble on|off")
    return
  end

  if cmd == "tooltip" then
    local v = (rest or ""):lower()
    if v == "on" then
      SIMCDPS_CONFIG.show_tooltip = true
      print("Mr. Mythical: DPS Predictor: tooltip output enabled")
      return
    end
    if v == "off" then
      SIMCDPS_CONFIG.show_tooltip = false
      print("Mr. Mythical: DPS Predictor: tooltip output disabled")
      return
    end
    if v == "status" or v == "" then
      local s = SIMCDPS_CONFIG.show_tooltip and "enabled" or "disabled"
      print("Mr. Mythical: DPS Predictor: tooltip output is " .. s)
      return
    end
    print("Mr. Mythical: DPS Predictor: /simcdps tooltip on|off|status")
    return
  end

  oldHandleSlash(msg)
end

-- Bag comparison UI
local BagComparisonUI = {}
NS.BagComparisonUI = BagComparisonUI
local bagComparisonFrame = nil
local bagComparisonData = {}
local bagComparisonRows = {}
local bagBestSetSummary = nil
local bagOverviewSummary = nil
local bagScanRunner = nil

local DEFAULT_BAG_SCAN_YIELD_EVERY = 40
local BAG_SCAN_SLOT_ORDER = { 16, 17, 1, 2, 3, 15, 5, 9, 10, 6, 7, 8, 11, 12 }
local BAG_NONWEAPON_SLOT_ORDER = { 1, 2, 3, 15, 5, 9, 10, 6, 7, 8, 11, 12 }

local SLOT_ID_LABELS = {
  [1] = "Head",
  [2] = "Neck",
  [3] = "Shoulder",
  [5] = "Chest",
  [6] = "Waist",
  [7] = "Legs",
  [8] = "Feet",
  [9] = "Wrist",
  [10] = "Hands",
  [11] = "Ring 1",
  [12] = "Ring 2",
  [13] = "Trinket 1",
  [14] = "Trinket 2",
  [15] = "Back",
  [16] = "Main Hand",
  [17] = "Off Hand",
}

local SLOT_ID_ORDER = {
  [1] = 1,
  [2] = 2,
  [3] = 3,
  [15] = 4,
  [5] = 5,
  [9] = 6,
  [10] = 7,
  [6] = 8,
  [7] = 9,
  [8] = 10,
  [11] = 11,
  [12] = 12,
  [13] = 13,
  [14] = 14,
  [16] = 15,
  [17] = 16,
}

local function getScenarioSlotKeyAndLabel(itemEquipLoc, p)
  if p and p.mode == "ring1" then
    return "slot_11", "Ring 1", SLOT_ID_ORDER[11]
  end
  if p and p.mode == "ring2" then
    return "slot_12", "Ring 2", SLOT_ID_ORDER[12]
  end
  if p and p.mode == "oh_replacement" then
    return "slot_17", "Off Hand", SLOT_ID_ORDER[17]
  end
  if p and (p.mode == "mh_replacement" or p.mode == "2h_replacement" or p.mode == "dw_pair_replacement") then
    return "slot_16", "Main Hand", SLOT_ID_ORDER[16]
  end
  if p and p.slot_id and SLOT_ID_LABELS[p.slot_id] then
    return "slot_" .. tostring(p.slot_id), SLOT_ID_LABELS[p.slot_id], SLOT_ID_ORDER[p.slot_id] or 999
  end

  local slots = INVTYPE_TO_SLOT_IDS[itemEquipLoc or ""]
  if slots and slots[1] and SLOT_ID_LABELS[slots[1]] then
    local sid = slots[1]
    return "slot_" .. tostring(sid), SLOT_ID_LABELS[sid], SLOT_ID_ORDER[sid] or 999
  end

  return (itemEquipLoc or "unknown"), (itemEquipLoc or "Unknown"), 999
end

local function makeZeroStats()
  return {
    primary_stat = 0,
    crit = 0,
    haste = 0,
    mastery = 0,
    versatility = 0,
  }
end

local ITEM_STAT_KEY_TO_FEATURE = {
  ITEM_MOD_STRENGTH_SHORT = "primary_stat",
  ITEM_MOD_AGILITY_SHORT = "primary_stat",
  ITEM_MOD_INTELLECT_SHORT = "primary_stat",
  ITEM_MOD_CRIT_RATING_SHORT = "crit",
  ITEM_MOD_CR_CRIT_SHORT = "crit",
  ITEM_MOD_HASTE_RATING_SHORT = "haste",
  ITEM_MOD_MASTERY_RATING_SHORT = "mastery",
  ITEM_MOD_VERSATILITY = "versatility",
  ITEM_MOD_VERSATILITY_SHORT = "versatility",
}

local function getItemStatVector(itemLink)
  local out = makeZeroStats()
  if not itemLink or not (C_Item and C_Item.GetItemStats) then
    return out
  end
  local ok, stats = pcall(C_Item.GetItemStats, itemLink)
  if not ok or type(stats) ~= "table" then
    return out
  end
  for k, v in pairs(stats) do
    if type(k) == "string" and type(v) == "number" then
      local feature = ITEM_STAT_KEY_TO_FEATURE[k]
      if feature then
        out[feature] = (out[feature] or 0) + v
      end
    end
  end
  return out
end

local function statsDiff(a, b)
  return {
    primary_stat = (a.primary_stat or 0) - (b.primary_stat or 0),
    crit = (a.crit or 0) - (b.crit or 0),
    haste = (a.haste or 0) - (b.haste or 0),
    mastery = (a.mastery or 0) - (b.mastery or 0),
    versatility = (a.versatility or 0) - (b.versatility or 0),
  }
end

local function isArmorCandidateAllowedForClass(classToken, itemClassID, itemSubClassID)
  if itemClassID ~= 4 then
    return true
  end
  if itemSubClassID == 0 or itemSubClassID == 6 then
    return true
  end
  local primaryArmor = CLASS_PRIMARY_ARMOR[classToken]
  return (not primaryArmor) or itemSubClassID == primaryArmor
end

local function buildCandidateKey(ref)
  if not ref then return nil end
  if ref.key then return ref.key end
  if ref.guid then return "guid:" .. tostring(ref.guid) end
  if ref.bag and ref.slot then return "bag:" .. tostring(ref.bag) .. ":" .. tostring(ref.slot) end
  if ref.slotId then return "eq:" .. tostring(ref.slotId) end
  if ref.link then return "link:" .. tostring(ref.link) end
  return nil
end

local function getItemScanSignals(itemLink)
  local ilvl = 0
  local getDetailedItemLevelInfo = rawget(_G, "GetDetailedItemLevelInfo")
  if getDetailedItemLevelInfo then
    local okIlvl, v = pcall(getDetailedItemLevelInfo, itemLink)
    if okIlvl and type(v) == "number" then
      ilvl = v
    end
  end

  if ilvl <= 0 then
    local _, _, _, fallbackIlvl = GetItemInfo(itemLink)
    if type(fallbackIlvl) == "number" then
      ilvl = fallbackIlvl
    end
  end

  local weaponDps = 0
  if C_Item and C_Item.GetItemStats then
    local okStats, stats = pcall(C_Item.GetItemStats, itemLink)
    if okStats and type(stats) == "table" then
      weaponDps = (stats.ITEM_MOD_DAMAGE_PER_SECOND_SHORT or 0) + (stats.ITEM_MOD_DAMAGE_PER_SECOND or 0)
    end
  end

  return ilvl or 0, weaponDps or 0
end

local function makeCandidateFromRef(ref)
  local link = itemRefToLink(ref)
  local guid = ref and (ref.guid or ref.itemGUID) or nil
  local name, _, quality = GetItemInfo(link)
  local _, _, equipLoc = getItemTypeInfo(link)
  local ilvl, weaponDps = getItemScanSignals(link)
  return {
    key = buildCandidateKey(ref),
    link = link,
    guid = guid,
    name = name or (link or "Unknown Item"),
    quality = quality or -1,
    equipLoc = equipLoc,
    stats = getItemStatVector(link),
    ilvl = ilvl,
    weapon_dps = weaponDps,
  }
end

local function makeEmptyCandidate(slotId)
  return {
    key = "empty:" .. tostring(slotId),
    link = nil,
    guid = nil,
    name = "(Empty)",
    quality = -1,
    equipLoc = nil,
    stats = makeZeroStats(),
    ilvl = 0,
    weapon_dps = 0,
  }
end

local function slotCanUseCandidate(slotId, cand, classToken, specKey, loadout)
  if not cand then return false end
  if not cand.link then
    return slotId == 17 or slotId == 11 or slotId == 12 or slotId == 13 or slotId == 14
  end

  local itemClassID, itemSubClassID, equipLoc = getItemTypeInfo(cand.link)
  if not equipLoc then return false end

  if slotId == 16 then
    if equipLoc == "INVTYPE_2HWEAPON" then
      return loadout.two_handed == true
    end
    return isItemAllowedForMainHand(classToken, itemClassID, itemSubClassID, equipLoc)
  end

  if slotId == 17 then
    if equipLoc == "INVTYPE_2HWEAPON" then
      return false
    end
    return isItemAllowedForOffHand(classToken, specKey, itemClassID, itemSubClassID, equipLoc)
  end

  if slotId == 11 or slotId == 12 then
    return equipLoc == "INVTYPE_FINGER"
  end

  if slotId == 13 or slotId == 14 then
    return equipLoc == "INVTYPE_TRINKET"
  end

  local slots = INVTYPE_TO_SLOT_IDS[equipLoc]
  if not slots then return false end
  local fits = false
  for _, sid in ipairs(slots) do
    if sid == slotId then
      fits = true
      break
    end
  end
  if not fits then return false end

  return isArmorCandidateAllowedForClass(classToken, itemClassID, itemSubClassID)
end

local function getBagItemSelectionTable()
  if type(SIMCDPS_CONFIG.bag_item_selection) ~= "table" then
    SIMCDPS_CONFIG.bag_item_selection = {}
  end
  return SIMCDPS_CONFIG.bag_item_selection
end

local function getBagSelectionKey(candidateOrKey)
  if type(candidateOrKey) == "table" then
    if candidateOrKey.guid then
      return "guid:" .. tostring(candidateOrKey.guid)
    end
    if candidateOrKey.link then
      return "link:" .. tostring(candidateOrKey.link)
    end
    return candidateOrKey.key
  end
  return candidateOrKey
end

local function isBagCandidateSelected(candidateOrKey)
  local key = getBagSelectionKey(candidateOrKey)
  if not key then
    return true
  end
  local selection = getBagItemSelectionTable()
  return selection[key] ~= false
end

local function setBagCandidateSelected(candidateOrKey, selected)
  local key = getBagSelectionKey(candidateOrKey)
  if not key then
    return
  end
  local selection = getBagItemSelectionTable()
  if selected then
    selection[key] = nil
  else
    selection[key] = false
  end
end

local function resetBagCandidateSelection()
  SIMCDPS_CONFIG.bag_item_selection = {}
end

local function collectBagSelectableItems(specKey)
  local out = {}
  if not specKey then
    return out
  end
  if not (C_Container and C_Container.GetContainerNumSlots and C_Container.GetContainerItemInfo) then
    return out
  end

  local _, classToken = UnitClass("player")
  local loadout = getWeaponLoadoutForSpec(specKey)
  local slotEnabled = {}
  for _, slotId in ipairs(BAG_SCAN_SLOT_ORDER) do
    slotEnabled[slotId] = true
  end
  local seen = {}

  for bag = 0, 4 do
    for slot = 1, C_Container.GetContainerNumSlots(bag) do
      local info = C_Container.GetContainerItemInfo(bag, slot)
      if info and info.hyperlink then
        local cand = makeCandidateFromRef({
          link = info.hyperlink,
          guid = info["itemGUID"] or info["guid"] or getGuidFromBagSlot(bag, slot),
          bag = bag,
          slot = slot,
        })

        if cand and cand.key and cand.equipLoc and cand.equipLoc ~= "" and cand.equipLoc ~= "INVTYPE_AMMO" then
          local candidateSlots = INVTYPE_TO_SLOT_IDS[cand.equipLoc]
          local usable = false
          if candidateSlots then
            for _, slotId in ipairs(candidateSlots) do
              if slotEnabled[slotId] and slotCanUseCandidate(slotId, cand, classToken, specKey, loadout) then
                usable = true
                break
              end
            end
          end

          if usable and not seen[cand.key] then
            seen[cand.key] = true
            cand.bag = bag
            cand.slot = slot
            table.insert(out, cand)
          end
        end
      end
    end
  end

  table.sort(out, function(a, b)
    local an = (a.name or a.link or ""):lower()
    local bn = (b.name or b.link or ""):lower()
    if an == bn then
      local ab = (a.bag or 0)
      local bb = (b.bag or 0)
      if ab == bb then
        return (a.slot or 0) < (b.slot or 0)
      end
      return ab < bb
    end
    return an < bn
  end)

  return out
end

local function isValidWeaponCombo(mhCand, ohCand, classToken, specKey, loadout, requireOffhandFor1H)
  if not mhCand or not mhCand.link then
    return false
  end

  local mhIs2H = is2HWeapon(mhCand.link)
  if mhIs2H then
    if not loadout.two_handed then
      return false
    end
    return (not ohCand) or (not ohCand.link)
  end

  local mhClassID, mhSubClassID, mhEquipLoc = getItemTypeInfo(mhCand.link)
  if not isItemAllowedForMainHand(classToken, mhClassID, mhSubClassID, mhEquipLoc) then
    return false
  end

  if ohCand and ohCand.link then
    if not loadout.dual_wield then
      return false
    end
    local ohClassID, ohSubClassID, ohEquipLoc = getItemTypeInfo(ohCand.link)
    return isItemAllowedForOffHand(classToken, specKey, ohClassID, ohSubClassID, ohEquipLoc)
  end

  if requireOffhandFor1H and loadout.dual_wield then
    return false
  end

  return true
end

local function setBagScanButtonState(isScanning)
  if not bagComparisonFrame or not bagComparisonFrame.scanBtn then
    return
  end
  if isScanning then
    bagComparisonFrame.scanBtn:SetText("Cancel")
  else
    bagComparisonFrame.scanBtn:SetText("Scan Bags")
  end
end

local function setBagScanStatusText(text)
  if bagComparisonFrame and bagComparisonFrame.summaryText then
    bagComparisonFrame.summaryText:SetFontObject(GameFontNormalSmall)
    bagComparisonFrame.summaryText:SetText(text)
  end
end

local function cancelBagScan()
  if not bagScanRunner then
    return
  end
  bagScanRunner.cancelled = true
  bagScanRunner = nil
  setBagScanButtonState(false)
  setBagScanStatusText("Best Set DPS: scan cancelled")
end

local function buildBagOverviewData()
  local specKey = SIMCDPS_CONFIG.spec_key or active_spec_keys[1]
  if not specKey then
    return nil, nil
  end

  local _, classToken = UnitClass("player")
  local loadout = getWeaponLoadoutForSpec(specKey)
  local slotOrder = BAG_SCAN_SLOT_ORDER
  local nonWeaponSlotOrder = BAG_NONWEAPON_SLOT_ORDER

  local slotCandidates = {}
  local slotSelectedCandidates = {}
  local slotSeen = {}
  local equippedBySlot = {}

  for _, slotId in ipairs(slotOrder) do
    slotCandidates[slotId] = {}
    slotSelectedCandidates[slotId] = {}
    slotSeen[slotId] = {}
    local eqRef = getSlotItemRef(slotId)
    local eqCand = eqRef and makeCandidateFromRef(eqRef) or makeEmptyCandidate(slotId)
    equippedBySlot[slotId] = eqCand
    if eqCand and eqCand.key then
      table.insert(slotCandidates[slotId], eqCand)
      table.insert(slotSelectedCandidates[slotId], eqCand)
      slotSeen[slotId][eqCand.key] = true
    end
  end

  local function addCandidateToSlot(slotId, cand)
    if not cand or not cand.key then return end
    if slotSeen[slotId][cand.key] then return end
    if not slotCanUseCandidate(slotId, cand, classToken, specKey, loadout) then return end

    table.insert(slotCandidates[slotId], cand)
    if isBagCandidateSelected(cand) then
      table.insert(slotSelectedCandidates[slotId], cand)
    end
    slotSeen[slotId][cand.key] = true
  end

  for bag = 0, 4 do
    for slot = 1, C_Container.GetContainerNumSlots(bag) do
      local info = C_Container.GetContainerItemInfo(bag, slot)
      if info and info.hyperlink then
        local cand = makeCandidateFromRef({
          link = info.hyperlink,
          guid = info["itemGUID"] or info["guid"] or getGuidFromBagSlot(bag, slot),
          bag = bag,
          slot = slot,
        })
        if cand and cand.equipLoc and cand.equipLoc ~= "" and cand.equipLoc ~= "INVTYPE_AMMO" then
          local candidateSlots = INVTYPE_TO_SLOT_IDS[cand.equipLoc]
          if candidateSlots then
            for _, slotId in ipairs(candidateSlots) do
              if slotCandidates[slotId] then
                addCandidateToSlot(slotId, cand)
              end
            end
          end
        end
      end
    end
  end

  if #slotCandidates[17] == 0 then
    local emptyOffhand = makeEmptyCandidate(17)
    table.insert(slotCandidates[17], emptyOffhand)
    slotSeen[17][emptyOffhand.key] = true
  end
  if #slotSelectedCandidates[17] == 0 then
    local emptyOffhand = makeEmptyCandidate(17)
    table.insert(slotSelectedCandidates[17], emptyOffhand)
  end

  local strictRequireOffhandFor1H = false
  for _, cand in ipairs(slotSelectedCandidates[17] or {}) do
    if cand and cand.link then
      strictRequireOffhandFor1H = true
      break
    end
  end

  local function countWeaponPairs(requireOffhandFor1H)
    local mhList = slotSelectedCandidates[16] or {}
    local ohList = slotSelectedCandidates[17] or {}
    local total = 0
    for _, mhCand in ipairs(mhList) do
      for _, ohCand in ipairs(ohList) do
        local mhKey = mhCand and mhCand.key or nil
        local ohKey = ohCand and ohCand.key or nil
        local ohIsEmpty = ohKey and ohKey:sub(1, 6) == "empty:"
        if ((not mhKey) or (not ohKey) or ohIsEmpty or mhKey ~= ohKey)
          and isValidWeaponCombo(mhCand, ohCand, classToken, specKey, loadout, requireOffhandFor1H) then
          total = total + 1
        end
      end
    end
    return total
  end

  local function countPairedSlotCombos(slotA, slotB, dedupeSwapped)
    local listA = slotSelectedCandidates[slotA] or {}
    local listB = slotSelectedCandidates[slotB] or {}
    if #listA == 0 or #listB == 0 then
      return 0
    end
    local n = 0
    for _, a in ipairs(listA) do
      for _, b in ipairs(listB) do
        local keyA = a and a.key or nil
        local keyB = b and b.key or nil
        local aEmpty = keyA and keyA:sub(1, 6) == "empty:"
        local bEmpty = keyB and keyB:sub(1, 6) == "empty:"
        if aEmpty or bEmpty or (keyA ~= keyB) then
          if dedupeSwapped and keyA and keyB and not aEmpty and not bEmpty and keyA > keyB then
            -- For paired slots (rings), only count one canonical ordering.
          else
            n = n + 1
          end
        end
      end
    end
    return n
  end

  local weaponPairs = countWeaponPairs(strictRequireOffhandFor1H)
  if weaponPairs == 0 and strictRequireOffhandFor1H then
    weaponPairs = countWeaponPairs(false)
  end

  local totalCombinations = weaponPairs
  local ringCombos = countPairedSlotCombos(11, 12, true)
  totalCombinations = totalCombinations * ringCombos

  for _, slotId in ipairs(nonWeaponSlotOrder) do
    if slotId ~= 11 and slotId ~= 12 then
      local list = slotSelectedCandidates[slotId] or {}
      local count = #list
      if count == 0 then
        totalCombinations = 0
        break
      end
      totalCombinations = totalCombinations * count
    end
  end

  local rows = {}
  for _, slotId in ipairs(slotOrder) do
    local equipped = equippedBySlot[slotId]
    local eqKey = equipped and equipped.key or nil
    local options = {}
    for _, cand in ipairs(slotCandidates[slotId] or {}) do
      local key = cand and cand.key or nil
      local isEquipped = eqKey and key == eqKey
      if cand and cand.link and not isEquipped then
        table.insert(options, cand)
      end
    end

    table.insert(rows, {
      row_type = "overview-slot",
      slot_order = SLOT_ID_ORDER[slotId] or 999,
      slot_label = SLOT_ID_LABELS[slotId] or tostring(slotId),
      equipped = equipped,
      options = options,
      is_upgrade = false,
    })
  end

  return rows, {
    spec_key = specKey,
    total_combinations = totalCombinations,
  }
end

local function refreshBagOverview()
  local rows, summary = buildBagOverviewData()
  bagBestSetSummary = nil
  bagComparisonData = rows or {}
  bagOverviewSummary = summary
  if summary then
    setBagScanStatusText(string.format(
      "Overview: %d total combinations (click icons to include/exclude, then Scan Bags)",
      summary.total_combinations or 0
    ))
  else
    setBagScanStatusText("Overview: missing spec key")
  end
  updateBagComparisonList()
end

local function syncBagUiConfigControls()
  if not bagComparisonFrame then return end
  if bagComparisonFrame.yieldEdit then
    bagComparisonFrame.yieldEdit:SetText(tostring(tonumber(SIMCDPS_CONFIG.bag_scan_yield_every) or DEFAULT_BAG_SCAN_YIELD_EVERY))
  end
end

local function scanBags()
  if bagScanRunner then
    return
  end

  bagComparisonData = {}
  bagOverviewSummary = nil
  bagBestSetSummary = nil

  local specKey = SIMCDPS_CONFIG.spec_key or active_spec_keys[1]
  if not specKey then
    return
  end

  local _, classToken = UnitClass("player")
  local loadout = getWeaponLoadoutForSpec(specKey)

  -- Put weapon slots first so 2H vs 1H+offhand is decided early.
  -- This avoids combination-cap bias from exploring thousands of armor/ring permutations
  -- before reaching main/offhand branches.
  local slotOrder = BAG_SCAN_SLOT_ORDER
  local slotCandidates = {}
  local slotSeen = {}
  local equippedBySlot = {}

  for _, slotId in ipairs(slotOrder) do
    slotCandidates[slotId] = {}
    slotSeen[slotId] = {}
    local eqRef = getSlotItemRef(slotId)
    local eqCand = eqRef and makeCandidateFromRef(eqRef) or makeEmptyCandidate(slotId)
    equippedBySlot[slotId] = eqCand
    if eqCand and eqCand.key then
      table.insert(slotCandidates[slotId], eqCand)
      slotSeen[slotId][eqCand.key] = true
    end
  end

  local function addCandidateToSlot(slotId, cand)
    if not cand or not cand.key then return end
    if slotSeen[slotId][cand.key] then return end
    if not slotCanUseCandidate(slotId, cand, classToken, specKey, loadout) then return end

    table.insert(slotCandidates[slotId], cand)
    slotSeen[slotId][cand.key] = true
  end
  
  for bag = 0, 4 do  -- 0=backpack, 1-4=bags
    for slot = 1, C_Container.GetContainerNumSlots(bag) do
      local info = C_Container.GetContainerItemInfo(bag, slot)
      if info and info.hyperlink then
        local cand = makeCandidateFromRef({
          link = info.hyperlink,
          guid = info["itemGUID"] or info["guid"] or getGuidFromBagSlot(bag, slot),
          bag = bag,
          slot = slot,
        })
        if cand and isBagCandidateSelected(cand) and cand.equipLoc and cand.equipLoc ~= "" and cand.equipLoc ~= "INVTYPE_AMMO" then
          local candidateSlots = INVTYPE_TO_SLOT_IDS[cand.equipLoc]
          if candidateSlots then
            for _, slotId in ipairs(candidateSlots) do
              if slotCandidates[slotId] then
                addCandidateToSlot(slotId, cand)
              end
            end
          end
        end
      end
    end
  end

  if #slotCandidates[17] == 0 then
    local emptyOffhand = makeEmptyCandidate(17)
    table.insert(slotCandidates[17], emptyOffhand)
    slotSeen[17][emptyOffhand.key] = true
  end

  local baseStats = getPlayerStatVector()
  local basePred = getCachedBaseDps(baseStats, specKey)

  local zeroStats = makeZeroStats()
  local deltaCacheBySlot = {}

  local function getCandidateDelta(slotId, cand)
    if not cand or not cand.key then
      return zeroStats
    end

    local slotCache = deltaCacheBySlot[slotId]
    if not slotCache then
      slotCache = {}
      deltaCacheBySlot[slotId] = slotCache
    end

    local cached = slotCache[cand.key]
    if cached then
      return cached
    end

    local equipped = equippedBySlot[slotId]
    local candStats = cand.stats or zeroStats
    local eqStats = (equipped and equipped.stats) or zeroStats
    cached = {
      primary_stat = (candStats.primary_stat or 0) - (eqStats.primary_stat or 0),
      crit = (candStats.crit or 0) - (eqStats.crit or 0),
      haste = (candStats.haste or 0) - (eqStats.haste or 0),
      mastery = (candStats.mastery or 0) - (eqStats.mastery or 0),
      versatility = (candStats.versatility or 0) - (eqStats.versatility or 0),
    }
    slotCache[cand.key] = cached
    return cached
  end

  local scoreStatsTmp = {
    primary_stat = 0,
    crit = 0,
    haste = 0,
    mastery = 0,
    versatility = 0,
  }

  -- Simple stat score used as a fallback/tie-breaker.
  local function candStatScore(cand)
    local s = cand.stats
    if not s then return 0 end
    return (s.primary_stat or 0) + (s.crit or 0) + (s.haste or 0)
           + (s.mastery or 0) + (s.versatility or 0)
  end

  local pruneScoreCache = {}

  local function candSearchScore(slotId, cand)
    if not cand or not cand.key then
      return -math.huge
    end

    local slotCache = pruneScoreCache[slotId]
    if not slotCache then
      slotCache = {}
      pruneScoreCache[slotId] = slotCache
    end

    local cached = slotCache[cand.key]
    if cached ~= nil then
      return cached
    end

    local score
    if not cand.link then
      score = -math.huge
    elseif slotId == 16 or slotId == 17 then
      -- Weapon pairs are evaluated later; use raw stats for ordering only.
      score = candStatScore(cand)
    else
      local d = getCandidateDelta(slotId, cand)
      scoreStatsTmp.primary_stat = (baseStats.primary_stat or 0) + (d.primary_stat or 0)
      scoreStatsTmp.crit = (baseStats.crit or 0) + (d.crit or 0)
      scoreStatsTmp.haste = (baseStats.haste or 0) + (d.haste or 0)
      scoreStatsTmp.mastery = (baseStats.mastery or 0) + (d.mastery or 0)
      scoreStatsTmp.versatility = (baseStats.versatility or 0) + (d.versatility or 0)
      score = predictWithStats(scoreStatsTmp, specKey) - basePred
    end

    slotCache[cand.key] = score
    return score
  end

  local function sortSlotCandidates(slotId)
    local list = slotCandidates[slotId]
    if not list or #list <= 1 then
      return
    end

    local equippedKey = equippedBySlot[slotId] and equippedBySlot[slotId].key or nil
    table.sort(list, function(a, b)
      local scoreA = candSearchScore(slotId, a)
      local scoreB = candSearchScore(slotId, b)
      if scoreA ~= scoreB then
        return scoreA > scoreB
      end

      local aIsEquipped = equippedKey and a.key == equippedKey or false
      local bIsEquipped = equippedKey and b.key == equippedKey or false
      if aIsEquipped ~= bIsEquipped then
        return aIsEquipped
      end

      return candStatScore(a) > candStatScore(b)
    end)
  end

  for _, slotId in ipairs(slotOrder) do
    sortSlotCandidates(slotId)
  end

  local strictRequireOffhandFor1H = false
  for _, cand in ipairs(slotCandidates[17] or {}) do
    if cand and cand.link then
      strictRequireOffhandFor1H = true
      break
    end
  end

  local nonWeaponSlotOrder = BAG_NONWEAPON_SLOT_ORDER
  local yieldEvery = tonumber(SIMCDPS_CONFIG.bag_scan_yield_every) or DEFAULT_BAG_SCAN_YIELD_EVERY
  if yieldEvery < 1 then
    yieldEvery = DEFAULT_BAG_SCAN_YIELD_EVERY
  end
  local runner = { cancelled = false }
  bagScanRunner = runner
  setBagScanButtonState(true)
  setBagScanStatusText("Best Set DPS: scanning... checked 0 combinations")

  local function buildWeaponPairs(requireOffhandFor1H)
    local pairs = {}
    local mhList = slotCandidates[16] or {}
    local ohList = slotCandidates[17] or {}

    for _, mhCand in ipairs(mhList) do
      for _, ohCand in ipairs(ohList) do
        local mhKey = mhCand and mhCand.key or nil
        local ohKey = ohCand and ohCand.key or nil
        local ohIsEmpty = ohKey and ohKey:sub(1, 6) == "empty:"
        if (not mhKey) or (not ohKey) or ohIsEmpty or mhKey ~= ohKey then
          if isValidWeaponCombo(mhCand, ohCand, classToken, specKey, loadout, requireOffhandFor1H) then
            table.insert(pairs, {
              mh = mhCand,
              oh = ohCand,
              score = candSearchScore(16, mhCand) + candSearchScore(17, ohCand),
            })
          end
        end
      end
    end

    table.sort(pairs, function(a, b)
      return a.score > b.score
    end)

    return pairs
  end

  local function runSearch(requireOffhandFor1H)
    local bestPredLocal = nil
    local bestAssignLocal = nil
    local checkedLocal = 0
    local usedKeys = {}
    local currentAssign = {}
    local evalStatsTmp = {
      primary_stat = 0,
      crit = 0,
      haste = 0,
      mastery = 0,
      versatility = 0,
    }

    local function maybeYield()
      if checkedLocal > 0 and (checkedLocal % yieldEvery) == 0 then
        coroutine.yield({
          kind = "progress",
          checked = checkedLocal,
        })
      end
    end

    local function evalCurrent()
      local mh = currentAssign[16]
      local oh = currentAssign[17]
      if not isValidWeaponCombo(mh, oh, classToken, specKey, loadout, requireOffhandFor1H) then
        return
      end

      local ring1 = currentAssign[11]
      local ring2 = currentAssign[12]
      local ring1Key = ring1 and ring1.key or nil
      local ring2Key = ring2 and ring2.key or nil
      if ring1Key and ring2Key and ring1Key > ring2Key then
        -- Canonicalise ring assignments so A/B and B/A are not both evaluated.
        return
      end

      local totalPrimary = 0
      local totalCrit = 0
      local totalHaste = 0
      local totalMastery = 0
      local totalVersatility = 0
      for _, slotId in ipairs(slotOrder) do
        local chosen = currentAssign[slotId] or equippedBySlot[slotId]
        local d = getCandidateDelta(slotId, chosen)
        totalPrimary = totalPrimary + (d.primary_stat or 0)
        totalCrit = totalCrit + (d.crit or 0)
        totalHaste = totalHaste + (d.haste or 0)
        totalMastery = totalMastery + (d.mastery or 0)
        totalVersatility = totalVersatility + (d.versatility or 0)
      end

      evalStatsTmp.primary_stat = (baseStats.primary_stat or 0) + totalPrimary
      evalStatsTmp.crit = (baseStats.crit or 0) + totalCrit
      evalStatsTmp.haste = (baseStats.haste or 0) + totalHaste
      evalStatsTmp.mastery = (baseStats.mastery or 0) + totalMastery
      evalStatsTmp.versatility = (baseStats.versatility or 0) + totalVersatility

      local pred = predictWithStats(evalStatsTmp, specKey)
      checkedLocal = checkedLocal + 1
      maybeYield()
      if not bestPredLocal or pred > bestPredLocal then
        bestPredLocal = pred
        bestAssignLocal = {}
        for _, slotId in ipairs(slotOrder) do
          bestAssignLocal[slotId] = currentAssign[slotId]
        end
      end
    end

    local weaponPairs = buildWeaponPairs(requireOffhandFor1H)
    if #weaponPairs == 0 then
      return nil, nil, 0, false
    end

    local function dfsNonWeapon(idx)
      if idx > #nonWeaponSlotOrder then
        evalCurrent()
        return
      end

      local slotId = nonWeaponSlotOrder[idx]
      local list = slotCandidates[slotId]
      if not list or #list == 0 then
        currentAssign[slotId] = makeEmptyCandidate(slotId)
        dfsNonWeapon(idx + 1)
        currentAssign[slotId] = nil
        return
      end

      for _, cand in ipairs(list) do
        local key = cand.key
        local isEmpty = key and key:sub(1, 6) == "empty:"
        if isEmpty or not usedKeys[key] then
          currentAssign[slotId] = cand
          if not isEmpty then usedKeys[key] = true end
          dfsNonWeapon(idx + 1)
          if not isEmpty then usedKeys[key] = nil end
          currentAssign[slotId] = nil
        end
      end
    end

    for _, pair in ipairs(weaponPairs) do

      currentAssign[16] = pair.mh
      currentAssign[17] = pair.oh

      local mhKey = pair.mh and pair.mh.key or nil
      local ohKey = pair.oh and pair.oh.key or nil
      local mhIsEmpty = mhKey and mhKey:sub(1, 6) == "empty:"
      local ohIsEmpty = ohKey and ohKey:sub(1, 6) == "empty:"

      if mhKey and not mhIsEmpty then
        usedKeys[mhKey] = true
      end
      if ohKey and not ohIsEmpty then
        usedKeys[ohKey] = true
      end

      dfsNonWeapon(1)

      coroutine.yield({
        kind = "progress",
        checked = checkedLocal,
      })

      if mhKey and not mhIsEmpty then
        usedKeys[mhKey] = nil
      end
      if ohKey and not ohIsEmpty then
        usedKeys[ohKey] = nil
      end

      currentAssign[16] = nil
      currentAssign[17] = nil
    end

    return bestPredLocal, bestAssignLocal, checkedLocal, false
  end

  local searchCo = coroutine.create(function()
    local bestPred, bestAssign, checked, wasCapped = runSearch(strictRequireOffhandFor1H)
    local usedRelaxedWeaponFallback = false

    if not bestAssign and strictRequireOffhandFor1H then
      local relaxedPred, relaxedAssign, relaxedChecked, relaxedWasCapped = runSearch(false)
      if relaxedAssign then
        bestPred = relaxedPred
        bestAssign = relaxedAssign
        checked = relaxedChecked
        wasCapped = relaxedWasCapped
        usedRelaxedWeaponFallback = true
      end
    end

    if not bestAssign then
      return {
        kind = "done",
        hasResult = false,
      }
    end

    for _, slotId in ipairs(slotOrder) do
      local chosen = bestAssign[slotId]
      if chosen and chosen.link then
        local equipped = equippedBySlot[slotId]
        local isUpgrade = true
        if equipped and equipped.key and chosen.key and equipped.key == chosen.key then
          isUpgrade = false
        end
        table.insert(bagComparisonData, {
          slot_order = SLOT_ID_ORDER[slotId] or 999,
          slot_label = SLOT_ID_LABELS[slotId] or tostring(slotId),
          name = chosen.name,
          link = chosen.link,
          quality = chosen.quality,
          is_upgrade = isUpgrade,
        })
      end
    end

    bagBestSetSummary = {
      dps_base = basePred,
      dps_new = bestPred,
      dps_delta = (bestPred or basePred) - basePred,
      combinations_checked = checked,
      combinations_capped = wasCapped,
      combinations_budget = nil,
      candidates_pruned = false,
      used_relaxed_weapon_fallback = usedRelaxedWeaponFallback,
      spec_key = specKey,
    }

    return {
      kind = "done",
      hasResult = true,
    }
  end)

  local function finishScan(cancelled, err)
    if bagScanRunner ~= runner then
      return
    end

    bagScanRunner = nil
    setBagScanButtonState(false)

    if cancelled then
      setBagScanStatusText("Best Set DPS: scan cancelled")
      return
    end

    if err then
      setBagScanStatusText("Best Set DPS: scan failed")
      print("Mr. Mythical: DPS Predictor: bag scan failed: " .. tostring(err))
      return
    end

    updateBagComparisonList()
    if bagBestSetSummary then
      local fallbackNote = bagBestSetSummary.used_relaxed_weapon_fallback and " (weapon fallback)" or ""
      print(string.format(
        "Mr. Mythical: DPS Predictor: best full set delta=%s across %d combinations%s",
        formatDelta(bagBestSetSummary.dps_delta or 0),
        bagBestSetSummary.combinations_checked or 0,
        fallbackNote
      ))
    else
      print("Mr. Mythical: DPS Predictor: no valid full-set combination found")
    end
  end

  local function pumpScan()
    if bagScanRunner ~= runner then
      return
    end
    if runner.cancelled then
      finishScan(true)
      return
    end

    local ok, payload = coroutine.resume(searchCo)
    if not ok then
      finishScan(false, payload)
      return
    end

    if coroutine.status(searchCo) == "dead" then
      finishScan(false)
      return
    end

    if type(payload) == "table" and payload.kind == "progress" then
      setBagScanStatusText(string.format(
        "Best Set DPS: scanning... checked %d combinations",
        payload.checked or 0
      ))
    end

    C_Timer.After(0, pumpScan)
  end

  C_Timer.After(0, pumpScan)
end

local function createBagComparisonFrame()
  if bagComparisonFrame then
    if not bagComparisonFrame.summaryText then
      local summaryText = bagComparisonFrame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
      summaryText:SetFontObject(GameFontNormalSmall)
      summaryText:SetPoint("TOP", bagComparisonFrame, "TOP", 0, -30)
      summaryText:SetTextColor(0.8, 0.95, 1)
      summaryText:SetText("Best Set DPS: scan to compute")
      bagComparisonFrame.summaryText = summaryText
    else
      bagComparisonFrame.summaryText:SetFontObject(GameFontNormalSmall)
    end
    bagComparisonFrame:Show()
    syncBagUiConfigControls()
    refreshBagOverview()
    return
  end
  
  bagComparisonFrame = CreateFrame("Frame", "SimcDpsBagComparisonFrame", UIParent, "BackdropTemplate")
  bagComparisonFrame:SetSize(900, 600)
  bagComparisonFrame:SetPoint("CENTER")
  bagComparisonFrame:SetBackdrop({
    bgFile = "Interface/Tooltips/UI-Tooltip-Background",
    edgeFile = "Interface/Tooltips/UI-Tooltip-Border",
    tile = true, tileSize = 16, edgeSize = 16,
    insets = { left = 4, right = 4, top = 4, bottom = 4 }
  })
  bagComparisonFrame:SetBackdropColor(0.1, 0.1, 0.1, 0.9)
  bagComparisonFrame:SetBackdropBorderColor(0.5, 0.5, 0.5, 1)
  
  -- Title
  local title = bagComparisonFrame:CreateFontString(nil, "OVERLAY")
  title:SetFont("Fonts/FRIZQT_.TTF", 14, "OUTLINE")
  title:SetPoint("TOP", bagComparisonFrame, "TOP", 0, -10)
  title:SetText("Mr. Mythical: DPS Predictor - Best Full Set")
  title:SetTextColor(1, 1, 0.5)

  local summaryText = bagComparisonFrame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
  summaryText:SetFontObject(GameFontNormalSmall)
  summaryText:SetPoint("TOP", bagComparisonFrame, "TOP", 0, -30)
  summaryText:SetTextColor(0.8, 0.95, 1)
  summaryText:SetText("Best Set DPS: scan to compute")
  bagComparisonFrame.summaryText = summaryText
  
  -- Close button
  local closeBtn = CreateFrame("Button", nil, bagComparisonFrame, "UIPanelCloseButton")
  closeBtn:SetPoint("TOPRIGHT", bagComparisonFrame, "TOPRIGHT", -5, -5)
  
  -- Scan button
  local scanBtn = CreateFrame("Button", nil, bagComparisonFrame, "GameMenuButtonTemplate")
  scanBtn:SetSize(80, 22)
  scanBtn:SetPoint("TOPLEFT", bagComparisonFrame, "TOPLEFT", 10, -35)
  scanBtn:SetText("Scan Bags")
  scanBtn:SetScript("OnClick", function()
    if bagScanRunner then
      cancelBagScan()
      print("Mr. Mythical: DPS Predictor: bag scan cancelled")
    else
      scanBags()
    end
  end)
  bagComparisonFrame.scanBtn = scanBtn

  local controlStartX = 100
  local controlY = -35

  local yieldLabel = bagComparisonFrame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
  yieldLabel:SetPoint("TOPLEFT", bagComparisonFrame, "TOPLEFT", controlStartX, controlY - 2)
  yieldLabel:SetText("Yield")
  yieldLabel:SetTextColor(0.9, 0.9, 0.9)

  local yieldEdit = CreateFrame("EditBox", nil, bagComparisonFrame, "InputBoxTemplate")
  yieldEdit:SetSize(54, 20)
  yieldEdit:SetPoint("LEFT", yieldLabel, "RIGHT", 4, 0)
  yieldEdit:SetAutoFocus(false)
  yieldEdit:SetNumeric(true)
  yieldEdit:SetMaxLetters(6)
  bagComparisonFrame.yieldEdit = yieldEdit

  local applySettingsBtn = CreateFrame("Button", nil, bagComparisonFrame, "GameMenuButtonTemplate")
  applySettingsBtn:SetSize(64, 22)
  applySettingsBtn:SetPoint("LEFT", yieldEdit, "RIGHT", 10, 0)
  applySettingsBtn:SetText("Apply")
  applySettingsBtn:SetScript("OnClick", function()
    local y = tonumber((yieldEdit:GetText() or ""):match("%d+"))

    if y and y >= 1 then
      SIMCDPS_CONFIG.bag_scan_yield_every = math.floor(y)
    end

    syncBagUiConfigControls()
    if not bagScanRunner then
      refreshBagOverview()
    end
    print(string.format(
      "Mr. Mythical: DPS Predictor: bag settings updated (yield=%d)",
      tonumber(SIMCDPS_CONFIG.bag_scan_yield_every) or DEFAULT_BAG_SCAN_YIELD_EVERY
    ))
  end)
  bagComparisonFrame.applySettingsBtn = applySettingsBtn
  
  -- Scroll frame for items
  local itemList = CreateFrame("Frame", nil, bagComparisonFrame, "BackdropTemplate")
  itemList:SetPoint("TOPLEFT", bagComparisonFrame, "TOPLEFT", 10, -92)
  itemList:SetSize(880, 498)
  
  -- Store references
  bagComparisonFrame.itemList = itemList
  
  -- Column headers
  local headerFrame = CreateFrame("Frame", nil, bagComparisonFrame, "BackdropTemplate")
  headerFrame:SetSize(880, 25)
  headerFrame:SetPoint("TOPLEFT", bagComparisonFrame, "TOPLEFT", 10, -65)
  headerFrame:SetBackdrop({
    bgFile = "Interface/Tooltips/UI-Tooltip-Background",
    edgeFile = "Interface/Tooltips/UI-Tooltip-Border",
    tile = true, tileSize = 16, edgeSize = 1,
  })
  headerFrame:SetBackdropColor(0.2, 0.2, 0.2, 0.8)
  headerFrame:SetBackdropBorderColor(0.4, 0.4, 0.4, 1)
  
  local slotHeader = headerFrame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
  slotHeader:SetFont("Fonts/FRIZQT_.TTF", 11, "OUTLINE")
  slotHeader:SetPoint("LEFT", headerFrame, "LEFT", 10, 0)
  slotHeader:SetText("Slot")
  slotHeader:SetTextColor(1, 1, 1)

  local itemHeader = headerFrame:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
  itemHeader:SetFont("Fonts/FRIZQT_.TTF", 11, "OUTLINE")
  itemHeader:SetPoint("LEFT", headerFrame, "LEFT", 190, 0)
  itemHeader:SetText("Item")
  itemHeader:SetTextColor(1, 1, 1)

  syncBagUiConfigControls()
  refreshBagOverview()
  
  return bagComparisonFrame
end

function updateBagComparisonList()
  if not bagComparisonFrame then return end
  
  local itemList = bagComparisonFrame.itemList

  if bagComparisonFrame.summaryText then
    bagComparisonFrame.summaryText:SetFontObject(GameFontNormalSmall)
    if bagBestSetSummary then
      local fallbackNote = bagBestSetSummary.used_relaxed_weapon_fallback and " (weapon fallback)" or ""
      bagComparisonFrame.summaryText:SetText(string.format(
        "Best Set DPS: %.0f -> %.0f (%s) | checked %d combinations%s",
        bagBestSetSummary.dps_base or 0,
        bagBestSetSummary.dps_new or 0,
        formatDelta(bagBestSetSummary.dps_delta or 0),
        bagBestSetSummary.combinations_checked or 0,
        fallbackNote
      ))
    elseif bagOverviewSummary then
      bagComparisonFrame.summaryText:SetText(string.format(
        "Overview: %d total combinations (click icons to include/exclude, then Scan Bags)",
        bagOverviewSummary.total_combinations or 0
      ))
    else
      bagComparisonFrame.summaryText:SetText("Best Set DPS: no valid combination")
    end
  end
  
  -- Clear existing rows.
  -- GetChild is not available on all client builds, so track rows explicitly.
  for _, row in ipairs(bagComparisonRows) do
    if row then
      row:Hide()
      row:SetParent(nil)
    end
  end
  bagComparisonRows = {}
  
  -- Add item rows
  local yOffset = 5
  for i, item in ipairs(bagComparisonData) do
    if item.row_type == "overview-slot" then
      local icons = {}
      if item.equipped then
        table.insert(icons, {
          candidate = item.equipped,
          is_equipped = true,
          is_selected = true,
        })
      end
      for _, cand in ipairs(item.options or {}) do
        table.insert(icons, {
          candidate = cand,
          is_equipped = false,
          is_selected = isBagCandidateSelected(cand),
        })
      end

      local iconSize = 28
      local iconSpacing = 4
      local iconsPerLine = 18
      local iconCount = #icons
      local lines = math.max(1, math.ceil(math.max(1, iconCount) / iconsPerLine))
      local rowHeight = math.max(32, lines * (iconSize + iconSpacing) + 6)

      local row = CreateFrame("Frame", nil, itemList, "BackdropTemplate")
      table.insert(bagComparisonRows, row)
      row:SetSize(860, rowHeight)
      row:SetPoint("TOPLEFT", itemList, "TOPLEFT", 0, -yOffset)
      row:SetBackdrop({
        bgFile = "Interface/Tooltips/UI-Tooltip-Background",
        tile = true, tileSize = 16,
      })

      local baseR, baseG, baseB = 0.1, 0.1, 0.1
      if i % 2 == 0 then
        baseR, baseG, baseB = 0.15, 0.15, 0.15
      end
      row:SetBackdropColor(baseR, baseG, baseB, 0.55)

      local slotText = row:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
      slotText:SetFontObject(GameFontNormalSmall)
      slotText:SetPoint("TOPLEFT", row, "TOPLEFT", 10, -8)
      slotText:SetText(item.slot_label or "")
      slotText:SetWidth(100)
      slotText:SetJustifyH("LEFT")
      slotText:SetTextColor(1, 1, 1)

      for iconIndex, iconInfo in ipairs(icons) do
        local col = (iconIndex - 1) % iconsPerLine
        local line = math.floor((iconIndex - 1) / iconsPerLine)
        local x = 120 + col * (iconSize + iconSpacing)
        local y = -4 - line * (iconSize + iconSpacing)
        local cand = iconInfo.candidate

        local iconBtn = CreateFrame("Button", nil, row, "BackdropTemplate")
        iconBtn:SetSize(iconSize, iconSize)
        iconBtn:SetPoint("TOPLEFT", row, "TOPLEFT", x, y)
        iconBtn:SetBackdrop({
          bgFile = "Interface/Buttons/WHITE8X8",
          edgeFile = "Interface/Tooltips/UI-Tooltip-Border",
          tile = true, tileSize = 8, edgeSize = 10,
          insets = { left = 2, right = 2, top = 2, bottom = 2 },
        })

        local borderR, borderG, borderB = 0.35, 0.35, 0.35
        local bgA = 0.25
        if iconInfo.is_equipped then
          borderR, borderG, borderB = 0.95, 0.82, 0.25
          bgA = 0.45
        elseif iconInfo.is_selected then
          borderR, borderG, borderB = 0.2, 0.85, 0.25
          bgA = 0.35
        else
          borderR, borderG, borderB = 0.7, 0.2, 0.2
          bgA = 0.18
        end
        iconBtn:SetBackdropColor(0, 0, 0, bgA)
        iconBtn:SetBackdropBorderColor(borderR, borderG, borderB, 1)

        local tex = iconBtn:CreateTexture(nil, "ARTWORK")
        tex:SetAllPoints(iconBtn)
        local iconTexture = cand and cand.link and GetItemIcon(cand.link) or nil
        if iconTexture then
          tex:SetTexture(iconTexture)
        end
        if not iconInfo.is_equipped and not iconInfo.is_selected then
          tex:SetVertexColor(0.45, 0.45, 0.45, 0.9)
        else
          tex:SetVertexColor(1, 1, 1, 1)
        end

        if iconInfo.is_equipped then
          local tag = iconBtn:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
          tag:SetPoint("CENTER", iconBtn, "CENTER", 0, 0)
          tag:SetText("E")
          tag:SetTextColor(1, 0.95, 0.5)
        end

        iconBtn:SetScript("OnClick", function()
          if iconInfo.is_equipped then
            if cand and cand.link then
              HandleModifiedItemClick(cand.link)
            end
            return
          end
          setBagCandidateSelected(cand, not isBagCandidateSelected(cand))
          if not bagScanRunner then
            refreshBagOverview()
          else
            updateBagComparisonList()
          end
        end)
        iconBtn:SetScript("OnEnter", function()
          if cand and cand.link then
            GameTooltip:SetOwner(iconBtn, "ANCHOR_RIGHT")
            GameTooltip:SetHyperlink(cand.link)
            GameTooltip:Show()
          end
        end)
        iconBtn:SetScript("OnLeave", function()
          GameTooltip:Hide()
        end)
      end

      yOffset = yOffset + rowHeight + 5
    else
    local row = CreateFrame("Button", nil, itemList, "BackdropTemplate")
    table.insert(bagComparisonRows, row)
    row:SetSize(860, 25)
    row:SetPoint("TOPLEFT", itemList, "TOPLEFT", 0, -yOffset)
    row:SetBackdrop({
      bgFile = "Interface/Tooltips/UI-Tooltip-Background",
      tile = true, tileSize = 16,
    })
    
    local baseR, baseG, baseB = 0.1, 0.1, 0.1
    if i % 2 == 0 then
      baseR, baseG, baseB = 0.15, 0.15, 0.15
    end
    if item.is_upgrade then
      -- Green tint for slots where best-set item is different from currently equipped.
      row:SetBackdropColor(baseR * 0.6, math.min(1, baseG + 0.18), baseB * 0.6, 0.7)
    else
      row:SetBackdropColor(baseR, baseG, baseB, 0.5)
    end

    local icon = row:CreateTexture(nil, "ARTWORK")
    icon:SetSize(18, 18)
    icon:SetPoint("LEFT", row, "LEFT", 190, 0)
    if item.link then
      local iconTexture = GetItemIcon(item.link)
      if iconTexture then
        icon:SetTexture(iconTexture)
      end
    end

    local slotText = row:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
    slotText:SetFontObject(GameFontNormalSmall)
    slotText:SetPoint("LEFT", row, "LEFT", 10, 0)
    slotText:SetText(item.slot_label or "")
    slotText:SetWidth(170)
    slotText:SetJustifyH("LEFT")
    if item.is_upgrade then
      slotText:SetTextColor(0.7, 1, 0.7)
    else
      slotText:SetTextColor(1, 1, 1)
    end
    
    -- Item name (clickable for comparison)
    local nameText = row:CreateFontString(nil, "OVERLAY", "GameFontNormalSmall")
    nameText:SetFontObject(GameFontNormalSmall)
    nameText:SetPoint("LEFT", row, "LEFT", 214, 0)
    nameText:SetText(item.name or item.link or "Unknown Item")
    nameText:SetWidth(500)
    nameText:SetJustifyH("LEFT")
    
    -- Apply color based on quality; unknown quality (e.g. -1) falls back to white.
    local r, g, b = 1, 1, 1
    local quality = tonumber(item.quality)
    if quality and quality >= 0 then
      if C_Item and C_Item.GetItemQualityColor then
        local qr, qg, qb = C_Item.GetItemQualityColor(quality)
        if qr and qg and qb then
          r, g, b = qr, qg, qb
        end
      elseif GetItemQualityColor then
        local qr, qg, qb = GetItemQualityColor(quality)
        if qr and qg and qb then
          r, g, b = qr, qg, qb
        end
      end
    end
    nameText:SetTextColor(r, g, b)
    
    row:SetScript("OnClick", function()
      if item.link then
        HandleModifiedItemClick(item.link)
      end
    end)
    row:SetScript("OnEnter", function()
      if item.link then
        GameTooltip:SetOwner(row, "ANCHOR_RIGHT")
        GameTooltip:SetHyperlink(item.link)
        GameTooltip:Show()
      end
    end)
    row:SetScript("OnLeave", function()
      GameTooltip:Hide()
    end)
    
    yOffset = yOffset + 30
    end
  end
  
  itemList:SetHeight(math.max(30, yOffset))
end

local oldHandleSlash2 = handleSlash
handleSlash = function(msg)
  local cmd, rest = msg:match("^(%S*)%s*(.-)$")
  cmd = (cmd or ""):lower()

  if cmd == "bagyield" then
    local v = (rest or ""):lower()
    if v == "" or v == "status" then
      local cur = tonumber(SIMCDPS_CONFIG.bag_scan_yield_every) or DEFAULT_BAG_SCAN_YIELD_EVERY
      print(string.format("Mr. Mythical: DPS Predictor: bag yield interval is %d evaluations", cur))
      return
    end
    local n = tonumber(v)
    if n and n >= 1 then
      SIMCDPS_CONFIG.bag_scan_yield_every = math.floor(n)
      print(string.format("Mr. Mythical: DPS Predictor: bag yield interval set to %d", SIMCDPS_CONFIG.bag_scan_yield_every))
    else
      print("Mr. Mythical: DPS Predictor: /simcdps bagyield <n>|status")
    end
    return
  end
  
  if cmd == "bags" or cmd == "compare" then
    createBagComparisonFrame()
    refreshBagOverview()
    return
  end
  
  if cmd == "bagclose" then
    if bagScanRunner then
      cancelBagScan()
    end
    if bagComparisonFrame then
      bagComparisonFrame:Hide()
    end
    return
  end
  
  oldHandleSlash2(msg)
end

SLASH_SIMCDPS1 = "/simcdps"
SlashCmdList.SIMCDPS = handleSlash

local f = CreateFrame("Frame")
f:RegisterEvent("PLAYER_LOGIN")
f:RegisterEvent("PLAYER_SPECIALIZATION_CHANGED")
f:RegisterEvent("PLAYER_EQUIPMENT_CHANGED")

f:SetScript("OnEvent", function(_, event)
  if event == "PLAYER_LOGIN" then
    detectAndCacheProfiles()
    local mode = "single"
    if Model.deployment and Model.deployment.mode == "ensemble" then
      mode = string.format("ensemble(%s,size=%d)", Model.deployment.strategy or "equal", Model.deployment.size or 0)
    end
    local n = #active_spec_keys
    if n > 1 then
      print(string.format("Mr. Mythical: DPS Predictor loaded [%s, %d profiles for %s]. /simcdps for commands.",
        mode, n, active_spec_prefix or "?"))
    else
      print("Mr. Mythical: DPS Predictor loaded [" .. mode .. "]. /simcdps for commands.")
    end
    if not SIMCDPS_CONFIG.show_tooltip then
      print("Mr. Mythical: DPS Predictor: tooltip output is disabled. Use /simcdps tooltip on")
    end
  elseif event == "PLAYER_SPECIALIZATION_CHANGED" then
    -- Clear any manual override so auto-detect picks up the new spec.
    SIMCDPS_CONFIG.spec_key = nil
    profileDetectionDoneRef[1] = false
    baseDpsCacheDirty = true
    if NS._clearPredictionCache then NS._clearPredictionCache() end
    detectAndCacheProfiles()
    local n = #active_spec_keys
    if n > 0 then
      print(string.format("Mr. Mythical: DPS Predictor: detected %d profile(s) for %s", n, active_spec_prefix or "?"))
      if n > 1 then
        for _, k in ipairs(active_spec_keys) do
          local label = active_spec_prefix and getProfileLabel(k, active_spec_prefix) or k
          print("  " .. label .. "  (" .. k .. ")")
        end
      end
    end
  elseif event == "PLAYER_EQUIPMENT_CHANGED" then
    -- Gear changed: invalidate base DPS and tooltip prediction caches.
    baseDpsCacheDirty = true
    if NS._clearPredictionCache then NS._clearPredictionCache() end
  end
end)

do
  -- Returns true only when GameTooltip is showing because the user is hovering a real UI element.
  local function isRealHoverTooltip(tooltip)
    if not tooltip or tooltip ~= GameTooltip then
      return false
    end
    -- Must be actually shown (visible to the player).
    if tooltip.IsShown and not tooltip:IsShown() then
      return false
    end
    if not tooltip.GetOwner then
      return false
    end
    local owner = tooltip:GetOwner()
    if not owner then
      return false
    end
    -- Owner must be under the mouse cursor right now.
    if owner.IsMouseOver then
      return owner:IsMouseOver()
    end
    local getMouseFocus = rawget(_G, "GetMouseFocus")
    local focus = getMouseFocus and getMouseFocus() or nil
    return focus == owner
  end

  -- Cache recent prediction results so tooltip rebuilds don't re-run ML inference.
  -- LRU ring buffer: keeps up to 64 recent items.
  local CACHE_SIZE = 64
  local predictionCache = {}   -- itemLink -> {lines=..., order=N}
  local cacheOrder = 0
  local cacheCount = 0

  local function evictOldestCache()
    if cacheCount <= CACHE_SIZE then return end
    local oldest_key, oldest_order = nil, math.huge
    for k, v in pairs(predictionCache) do
      if v.order < oldest_order then
        oldest_key = k
        oldest_order = v.order
      end
    end
    if oldest_key then
      predictionCache[oldest_key] = nil
      cacheCount = cacheCount - 1
    end
  end

  local function clearPredictionCache()
    predictionCache = {}
    cacheCount = 0
    cacheOrder = 0
  end

  -- Expose so the event handler can invalidate on gear/spec change.
  NS._clearPredictionCache = clearPredictionCache

  local function getTooltipItemLink(tooltip)
    if not tooltip or not tooltip.GetItem then
      return nil
    end
    local ok, _, link = pcall(tooltip.GetItem, tooltip)
    if ok and link then
      return link
    end
    return nil
  end

  local function addPredictionLinesToTooltip(tooltip, itemLink, itemGuid, itemData)
    if not SIMCDPS_CONFIG.show_tooltip then
      return
    end
    if not tooltip or not itemLink or itemLink == "" then
      return
    end

    -- Skip items the player cannot equip (wrong armor type, class restriction, etc.).
    local isEquippable = rawget(_G, "IsEquippableItem")
    if isEquippable and not isEquippable(itemLink) then
      return
    end

    -- Get item classification to filter trinkets and wrong armor types.
    local itemClassID, itemSubClassID, equipLoc
    if C_Item and C_Item.GetItemInfoInstant then
      local _, _, _, loc, _, cid, sid = C_Item.GetItemInfoInstant(itemLink)
      itemClassID, itemSubClassID, equipLoc = cid, sid, loc
    else
      local _, _, _, _, _, _, _, _, loc, _, _, cid, sid = GetItemInfo(itemLink)
      itemClassID, itemSubClassID, equipLoc = cid, sid, loc
    end

    -- Skip trinkets — model uses BiS trinkets as baseline; stat-swap predictions are meaningless.
    if equipLoc == "INVTYPE_TRINKET" then
      return
    end

    -- Skip armor pieces that don't match the player's primary armor type.
    -- subClassID 0 = misc (neck, ring, cloak) and 6 = shield — always allow those.
    if itemClassID == 4 and itemSubClassID and itemSubClassID ~= 0 and itemSubClassID ~= 6 then
      local _, classToken = UnitClass("player")
      local primaryArmor = CLASS_PRIMARY_ARMOR[classToken]
      if primaryArmor and itemSubClassID ~= primaryArmor then
        return
      end
    end

    local ok, err = pcall(function()
      -- Use cached result if available for this exact item link.
      local cached = predictionCache[itemLink]
      if cached then
        cached.order = cacheOrder  -- refresh LRU position
        cacheOrder = cacheOrder + 1
        for _, line in ipairs(cached.lines) do
          tooltip:AddLine(line.text, line.r, line.g, line.b)
        end
        return
      end

      -- Attempt profile detection once if not yet done.
      if not profileDetectionDoneRef[1] and not SIMCDPS_CONFIG.spec_key and #active_spec_keys == 0 then
        detectAndCacheProfiles()
      end

      local specKeys
      if SIMCDPS_CONFIG.spec_key then
        specKeys = { SIMCDPS_CONFIG.spec_key }
      elseif #active_spec_keys > 0 then
        specKeys = active_spec_keys
      end
      if not specKeys or #specKeys == 0 then
        return
      end

      -- Defer inference to next frame to avoid blocking the tooltip render.
      -- The tooltip will appear normally, then prediction lines are appended one frame later.
      local capturedSpecKeys = specKeys
      C_Timer.After(0, function()
        -- Re-check: tooltip may have changed or hidden since we scheduled.
        if not tooltip:IsShown() then return end
        local currentLink
        if tooltip.GetItem then
          local ok2, _, link = pcall(tooltip.GetItem, tooltip)
          if ok2 then currentLink = link end
        end
        if currentLink ~= itemLink then return end

        local lines = {}
        for _, specKey in ipairs(capturedSpecKeys) do
          local pred = Predictor.PredictItemDelta({ link = itemLink, guid = itemGuid, comparisonItem = (type(itemData) == "table" and (itemData.item or itemData)) or nil }, specKey)
          if pred then
            -- Handle multiple results (weapons, rings, etc.)
            if type(pred) == "table" and pred[1] then
              -- Array of results
              for _, p in ipairs(pred) do
                local deltaText = formatDelta(p.dps_delta)
                local candidateR, candidateG, candidateB = 1, 0.2, 0.2
                if p.dps_delta >= 0 then
                  candidateR, candidateG, candidateB = 0.2, 1, 0.2
                end
                local label = (active_spec_prefix and getProfileLabel(specKey, active_spec_prefix)) or specKey
                local modeStr = ""
                if p.mode == "mh_replacement" then
                  modeStr = " (mainhand)"
                elseif p.mode == "2h_replacement" then
                  modeStr = " (2H)"
                elseif p.mode == "dw_pair_replacement" then
                  modeStr = " (paired 1H set)"
                elseif p.mode == "oh_replacement" then
                  modeStr = " (offhand)"
                elseif p.mode == "ring1" then
                  modeStr = " (ring 1)"
                elseif p.mode == "ring2" then
                  modeStr = " (ring 2)"
                end
                table.insert(lines, {
                  text = string.format("ML DPS [%s]%s: %s", label, modeStr, deltaText),
                  r = candidateR, g = candidateG, b = candidateB,
                })
              end
            else
              -- Single result
              local deltaText = formatDelta(pred.dps_delta)
              local candidateR, candidateG, candidateB = 1, 0.2, 0.2
              if pred.dps_delta >= 0 then
                candidateR, candidateG, candidateB = 0.2, 1, 0.2
              end
              local label = (active_spec_prefix and getProfileLabel(specKey, active_spec_prefix)) or specKey
              table.insert(lines, {
                text = string.format("ML DPS [%s]: %s", label, deltaText),
                r = candidateR, g = candidateG, b = candidateB,
              })
            end
          end
        end

        if #lines > 0 then
          -- Store in LRU cache.
          if not predictionCache[itemLink] then
            cacheCount = cacheCount + 1
          end
          predictionCache[itemLink] = { lines = lines, order = cacheOrder }
          cacheOrder = cacheOrder + 1
          evictOldestCache()

          tooltip:AddLine(" ")
          for _, line in ipairs(lines) do
            tooltip:AddLine(line.text, line.r, line.g, line.b)
          end
          tooltip:Show()  -- resize tooltip to fit new lines
        end
      end)
    end)

    if not ok and not didWarnTooltipError then
      didWarnTooltipError = true
      print("Mr. Mythical: DPS Predictor: tooltip prediction error: " .. tostring(err))
    end
  end

  -- Primary hook: TooltipDataProcessor fires AFTER item data is populated and provides the
  -- item link via data.hyperlink.  This is the only reliable path on modern retail clients
  -- (OnTooltipSetItem does not exist on GameTooltip; OnShow fires before item data is ready).
  local hooked = false
  if TooltipDataProcessor and TooltipDataProcessor.AddTooltipPostCall and Enum and Enum.TooltipDataType then
    TooltipDataProcessor.AddTooltipPostCall(Enum.TooltipDataType.Item, function(tooltip, data)
      if not tooltip then
        return
      end

      -- Strict filter: primary item tooltips + comparison tooltips.
      if tooltip == GameTooltip then
        -- Must be a real mouse-hover, not an internal tooltip query (character panel, comparisons).
        if not isRealHoverTooltip(tooltip) then
          return
        end
      elseif tooltip ~= ItemRefTooltip then
        -- Skip other tooltip frames.
        return
      end

      -- Extract item link from the data the processor provides.
      local itemLink = data and data.hyperlink
      if not itemLink and tooltip.GetItem then
        local ok2, _, link = pcall(tooltip.GetItem, tooltip)
        if ok2 and link then
          itemLink = link
        end
      end

      if itemLink then
        local itemGuid = type(data) == "table" and (data["guid"] or data["itemGUID"]) or nil
        addPredictionLinesToTooltip(tooltip, itemLink, itemGuid, data)
      end
    end)
    hooked = true
  end

  if not hooked then
    print("Mr. Mythical: DPS Predictor: TooltipDataProcessor unavailable – tooltip predictions disabled.")
  else
    print("Mr. Mythical: DPS Predictor: tooltip hooks active (TooltipDataProcessor)")
  end
end
