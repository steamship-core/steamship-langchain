# Testing

This library contains a suite of end-to-end tests that you can perform against Steamship.

If you are running a private-hosted version of Steamship, we recommend running this test suite as part of your setup, and then regularly to monitor system health.

## 1. Set up your Python Virtualenv

Follow the instructions in [DEVELOPING](DEVELOPING.md) to set up your virtual environment

## 3. Set Environment Variables

Set the following environment variables for testing:

1. **Your Steamship API key**.

```
export STEAMSHIP_API_KEY=
```

## 4. Run the tests

With the virtual environment active and environment variables set, run:

```
pytest
```