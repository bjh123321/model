import argparse

from surrogate_model.utils.argparse_compat import (
    BooleanOptionalAction,
    _BooleanOptionalActionFallback,
)


def _parse(action, argv, default=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", action=action, default=default)
    return parser.parse_args(argv).feature


def test_fallback_boolean_optional_action_values():
    assert _parse(_BooleanOptionalActionFallback, []) is None
    assert _parse(_BooleanOptionalActionFallback, ["--feature"]) is True
    assert _parse(_BooleanOptionalActionFallback, ["--no-feature"]) is False


def test_fallback_boolean_optional_action_preserves_explicit_default():
    assert _parse(_BooleanOptionalActionFallback, [], default=False) is False
    assert _parse(_BooleanOptionalActionFallback, ["--feature"], default=False) is True
    assert _parse(_BooleanOptionalActionFallback, ["--no-feature"], default=True) is False


def test_public_boolean_optional_action_has_matching_behavior():
    assert _parse(BooleanOptionalAction, []) is None
    assert _parse(BooleanOptionalAction, ["--feature"]) is True
    assert _parse(BooleanOptionalAction, ["--no-feature"]) is False
