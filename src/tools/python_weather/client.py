import requests
from requests.exceptions import RequestException as ClientResponseError
from urllib.parse import quote_plus
from typing import Optional
import time

from .errors import Error, RequestError
from .constants import _Unit, METRIC
from .base import CustomizableBase
from .forecast import Forecast
from .version import VERSION
from .enums import Locale


class Client(CustomizableBase):
    """
    Interact with the API's endpoints (Synchronous Version).
    """

    __slots__: tuple[str, ...] = ('__own_session', '__session', '__max_retries')

    def __init__(
        self,
        *,
        unit: _Unit = METRIC,
        locale: Locale = Locale.ENGLISH,
        session: Optional[requests.Session] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(unit, locale)

        self.__own_session = session is None
        self.__session = session or requests.Session()
        self.__session.headers.update({
            'User-Agent': f'python_weather (https://github.com/null8626/python-weather {VERSION}) Python/requests',
            'Content-Type': 'application/json',
        })
        self.__max_retries = max_retries or 3

    def __repr__(self) -> str:
        return f'<{__class__.__name__} session_id={id(self.__session)}>'

    def get(
        self,
        location: str,
        *,
        unit: Optional[_Unit] = None,
        locale: Optional[Locale] = None,
    ) -> Forecast:
        """
        Fetches a weather forecast for a specific location (Synchronous).
        """
        if not isinstance(location, str) or not location:
            raise TypeError(f'Expected a proper location str, got {location!r}')

        current_unit = unit if isinstance(unit, _Unit) else self._CustomizableBase__unit
        current_locale = locale if isinstance(locale, Locale) else self._CustomizableBase__locale

        subdomain = f'{current_locale.value}.' if current_locale != Locale.ENGLISH else ''
        delay = 0.5
        attempts = 0
        last_exception = None

        while attempts <= self.__max_retries or self.__max_retries == -1:
            try:
                resp = self.__session.get(
                    f'https://{subdomain}wttr.in/{quote_plus(location)}?format=j1',
                    timeout=10
                )
                resp.raise_for_status()
                return Forecast(resp.json(), current_unit, current_locale)
            except ClientResponseError as e:
                last_exception = e
                status = e.response.status_code if e.response is not None else None
                if self.__max_retries != -1 and attempts == self.__max_retries:
                    raise RequestError(status) from e

                time.sleep(delay)
                attempts += 1
                delay *= 2

        raise RequestError(getattr(last_exception, 'response', None) and last_exception.response.status_code) from last_exception

    def close(self) -> None:
        """Closes the session if it was created by this client."""
        if self.__own_session:
            self.__session.close()

    def __enter__(self) -> 'Client':
        return self

    def __exit__(self, *_, **__) -> None:
        self.close()