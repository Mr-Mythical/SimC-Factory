"""Generate a self-signed CA and client certificate for IAM Roles Anywhere.

Usage:
    python scripts/generate_certs.py

Creates files in certs/:
    ca.pem          - CA certificate (upload to Terraform as dashboard_ca_certificate_pem)
    ca-key.pem      - CA private key (keep safe, only needed to issue new client certs)
    client.pem      - Client certificate (used by aws_signing_helper)
    client-key.pem  - Client private key (used by aws_signing_helper)

The CA cert is valid for 10 years, client cert for 1 year.
"""

import os
import sys
from datetime import UTC, datetime, timedelta

# Ensure cryptography is available
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError:
    print("Missing dependency. Install with: pip install cryptography")
    sys.exit(1)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    certs_dir = os.path.join(project_root, "certs")
    os.makedirs(certs_dir, exist_ok=True)

    ca_cert_path = os.path.join(certs_dir, "ca.pem")
    ca_key_path = os.path.join(certs_dir, "ca-key.pem")
    client_cert_path = os.path.join(certs_dir, "client.pem")
    client_key_path = os.path.join(certs_dir, "client-key.pem")

    # Check if certs already exist
    if os.path.exists(ca_cert_path):
        print(f"Certificates already exist in {certs_dir}/")
        print("Delete the certs/ directory first if you want to regenerate.")
        sys.exit(0)

    now = datetime.now(UTC)

    # ── Generate CA key and self-signed certificate ─────────────────────────
    print("Generating CA key pair...")
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Mr. Mythical SimC Factory Dashboard CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Mr. Mythical"),
    ])

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=3650))  # 10 years
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA384())
    )

    # ── Generate client key and certificate signed by CA ────────────────────
    print("Generating client key pair...")
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    client_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "mr-mythical-simc-factory-dashboard"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Mr. Mythical"),
    ])

    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_name)
        .issuer_name(ca_name)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))  # 1 year
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA384())
    )

    # ── Write files ─────────────────────────────────────────────────────────
    with open(ca_key_path, "wb") as f:
        f.write(ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))

    with open(ca_cert_path, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    with open(client_key_path, "wb") as f:
        f.write(client_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))

    with open(client_cert_path, "wb") as f:
        f.write(client_cert.public_bytes(serialization.Encoding.PEM))

    print(f"\nCertificates generated in {certs_dir}/:")
    print("  ca.pem          — CA certificate (10 year validity)")
    print("  ca-key.pem      — CA private key (keep safe!)")
    print("  client.pem      — Client certificate (1 year validity)")
    print("  client-key.pem  — Client private key")
    print()
    print("Next steps:")
    print("  1. Add CA cert to Terraform:")
    print('     dashboard_ca_certificate_pem = file("../certs/ca.pem")')
    print("  2. Run: terraform apply")
    print("  3. Install aws_signing_helper:")
    print("     https://docs.aws.amazon.com/rolesanywhere/latest/userguide/credential-helper.html")
    print("  4. Update .env with the ARNs from terraform output and cert paths")


if __name__ == "__main__":
    main()
