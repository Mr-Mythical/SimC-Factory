from web import credentials


def test_saved_credentials_are_filtered_and_mapped_to_aws_env(tmp_path, monkeypatch):
    credentials_file = tmp_path / ".credentials.json"
    monkeypatch.setattr(credentials, "_CREDENTIALS_FILE", str(credentials_file))

    credentials.save({
        "aws_access_key_id": "  TESTKEY  ",
        "aws_secret_access_key": "  TESTSECRET  ",
        "aws_session_token": "",
        "aws_region": " us-east-1 ",
        "ignored": "value",
    })

    assert credentials.load() == {
        "aws_access_key_id": "TESTKEY",
        "aws_secret_access_key": "TESTSECRET",
        "aws_region": "us-east-1",
    }
    assert credentials.get_aws_env() == {
        "AWS_ACCESS_KEY_ID": "TESTKEY",
        "AWS_SECRET_ACCESS_KEY": "TESTSECRET",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
    }

    credentials.save({})
    assert credentials.load() == {}
    assert not credentials_file.exists()