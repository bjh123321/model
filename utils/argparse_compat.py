"""Compatibility helpers for argparse features added after Python 3.8."""

import argparse


class _BooleanOptionalActionFallback(argparse.Action):
    """Python 3.8 implementation matching argparse.BooleanOptionalAction."""

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        expanded_option_strings = []
        for option_string in option_strings:
            expanded_option_strings.append(option_string)
            if option_string.startswith("--"):
                expanded_option_strings.append("--no-" + option_string[2:])

        super().__init__(
            option_strings=expanded_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not option_string.startswith("--no-"))

    def format_usage(self):
        return " | ".join(self.option_strings)


BooleanOptionalAction = getattr(
    argparse, "BooleanOptionalAction", _BooleanOptionalActionFallback
)

