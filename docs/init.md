# Init project and env

## Create .venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Magentic-One uses playwright to interact with web pages. You need to install the playwright dependencies. Run the following command to install the playwright dependencies:

```bash
playwright install --with-deps chromium
```
