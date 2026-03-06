from . import Source as BaseSource


class Source(BaseSource):
    _id = "WB_WDI"

    def modify_request_args(self, kwargs):
        """World Bank's agency ID."""
        super().modify_request_args(kwargs)

        if kwargs["resource_type"] == "categoryscheme":
            # Service does not respond to requests for "WB" category schemes
            kwargs["provider"] = "all"
        elif kwargs["resource_type"] != "data":
            # Provider's own ID differs from its ID in this package
            kwargs.setdefault("provider", "WB")

        try:
            if isinstance(kwargs["key"], str):
                # Data queries fail without a trailing slash
                # TODO improve the hook integration with Client.get so this can be done
                #      after the query URL is prepared
                kwargs["key"] += "/"
        except KeyError:
            pass
