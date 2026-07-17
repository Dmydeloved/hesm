Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

if (-not $env:DASHSCOPE_API_KEY) {
    throw "Missing DASHSCOPE_API_KEY. Set it before API smoke."
}
if (-not $env:EVALUATION_API_KEY) {
    throw "Missing EVALUATION_API_KEY. Set it before API smoke."
}

python -m py_compile experiments\runner.py experiments\llm.py experiments\methods\hesm.py
python -m unittest tests.test_experiments -v
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s evaluation\tests -p "test_*.py"
python -m experiments.runner --config experiments\configs\multiwoz_smoke.yaml
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml
python -m experiments.aggregate --root results\experiments --output results\experiments\summary.csv

Write-Host "Preflight complete. Inspect results\experiments\multiwoz_api_smoke_hesm before the full run."

