from requests import HTTPError
from requests.exceptions import SSLError

__all__ = [
    "HTTPError",
    "ParseError",
    "SSLError",
    "XMLParseError",
]


class ParseError(Exception):
    """:class:`~.reader.Reader` is unable to parse a message."""


class XMLParseError(Exception):
    """:class:`.xml.Reader` is unable to parse a message."""

    def __str__(self):
        c = str(self.__cause__)
        return f"{self.__cause__.__class__.__name__}{': ' + c if c else ''}"
