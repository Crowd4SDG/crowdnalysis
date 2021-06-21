import re

from .. import _version as v

# Semantic Versioning Official Regular Expression: https://regex101.com/r/vkijKf/1/
OFFICIAL_SEMVER_REGEX = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"


def test_version_no():
    assert "__version__" in dir(v)
    assert re.match(OFFICIAL_SEMVER_REGEX, v.__version__)
