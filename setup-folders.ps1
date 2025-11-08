# setup-folders.ps1 — scaffold memory + sources inside allie-ai

# Set your project root to the allie-ai folder
$projectRoot = "C:\Users\brock\allieai\allie-ai"

# Ensure the base folder exists
if (-not (Test-Path $projectRoot)) {
    Write-Error "Project root not found: $projectRoot"
    exit 1
}

# Define the folder structure
$folders = @(
    "$projectRoot\memory",
    "$projectRoot\memory\tests",
    "$projectRoot\sources"
)

# Create each folder if it doesn't already exist
foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created folder: $folder"
    } else {
        Write-Host "Folder already exists: $folder"
    }
}

# Starter files
$files = @(
    "$projectRoot\memory\__init__.py",
    "$projectRoot\memory\db.py",
    "$projectRoot\memory\linked_list.py",
    "$projectRoot\memory\index.py",
    "$projectRoot\memory\hybrid.py",
    "$projectRoot\sources\duckduckgo.py",
    "$projectRoot\sources\wikidata.py",
    "$projectRoot\sources\dbpedia.py",
    "$projectRoot\sources\openlibrary.py",
    "$projectRoot\README.md"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file | Out-Null
        Write-Host "Created file: $file"
    } else {
        Write-Host "File already exists: $file"
    }
}
