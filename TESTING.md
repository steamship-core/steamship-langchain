# Testing

This library contains a suite of end-to-end tests that you can perform against Steamship.

If you are running a private-hosted version of Steamship, we recommend running this test suite as part of your setup, and then regularly to monitor system health.

## 1. Create a Test User

We recommend creating a test user to isolate any artifacts and workload that testing involves.

## 2. Set up your Python Virtualenv

Follow the instructions in [DEVELOPING](DEVELOPING.md) to set up your virtual environment

## 3. Set Environment Variables

Set the following environment variables for testing:

1. **Your Steamship API Domain**. If you are using `steamship.com`, this step is not necessary. If you have a private Steamship installation, use the API domain you normally use.

```
export STEAMSHIP_DOMAIN=https://api.steamship.yourcompany.com/
export STEAMSHIP_DOMAIN=http://localhost:8080
export STEAMSHIP_DOMAIN=https://api.staging.steamship.com/
```


2. **Your Steamship API key**.

```
export STEAMSHIP_KEY=
```

3. **Your default QA Embedding Model name**. For private installations, this default plugin may be custom. 

```
export STEAMSHIP_EMBEDDER_QA=st_msmarco_distilbert_base_v3
```

4. **Your default Similarity Embedding Model name**. For private installations, this default plugin may be custom. 

```
export STEAMSHIP_EMBEDDER_SIM=st_paraphrase_mpnet_base_v2
```

5. **Your default Parsing Model name**. For private installations, this default plugin may be custom. 

```
export STEAMSHIP_PARSER_DEFAULT=sp_en_core_web_trf
```

## 4. Run the tests

With the virtual environment active and environment variables set, run:

```
./bin/tox
```

Or run one test

```
./bin/tox -- tests/test_embedding_index.py::test_empty_queries
```


# Testing against Steamship on Localhost

Testing locally requires a few steps:

1. **Create a publicly accessible inbound proxy.** If using the cloud task scheduler, this will enable it to contact the Steamship Engine on the local machine.

    ```
    ngrok http 8080
    ```

2. **Wire up the inbound proxy.** 

    In your `~/.steamship-config.json` file, set the `queueUrl` with the NGrok URL just generated.

    **You must use the https variant!**

    ```
      "queueUrl": "https://a5c6eb28c411.ngrok.io/...",
    ```

3. **Run the Steamship Engine.** Await its availability on Port 8080.

4. **Run the steps for Production Testing**. See above.
