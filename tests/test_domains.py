import pytest

from dgalens.data import read_labels
from dgalens.domains import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "example"),
        ("WWW.Example.COM.", "example"),
        ("mail.example.co.uk", "example"),
        ("https://sub.example.org:8443/path?q=1", "example"),
        ("  example.net  ", "example"),
        ("com", ""),
        ("", ""),
        ("192.168.10.24", ""),
        ("http://10.0.0.1:8080/admin", ""),
        ("2001:db8::1", ""),
        ("[2001:db8::1]:443", ""),
        ("https://[2001:db8::1]/path", ""),
        ("1.2.3.4.example.com", "example"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_read_labels_handles_comments_blanks_and_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "domains.txt"
    path.write_text("# header\n\n1,google.com\n2,bbc.co.uk\nexample.org\ncom\n", encoding="utf-8")
    assert read_labels(path) == ["google", "bbc", "example"]
