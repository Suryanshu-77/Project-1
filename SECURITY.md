# Security policy

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/tierwise/) receives security fixes.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through
[GitHub's vulnerability reporting](https://github.com/Suryanshu-77/tierwise/security/advisories/new)
with the affected version, steps to reproduce and the impact you expect.

You can expect an acknowledgement within 7 days. Once a fix is released, the advisory is
published with credit to the reporter unless you ask otherwise.

## Scope

tierwise holds provider API keys and, when payload storage is enabled, request text. Of
particular interest:

- bypassing API-key authentication, budgets or rate limits
- leaking provider keys, tierwise keys or stored payloads (logs, errors, metrics, traces)
- a request that makes the gateway call a host other than a configured provider
