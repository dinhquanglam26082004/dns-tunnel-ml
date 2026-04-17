# PowerShell Workflow: Upgrade to Gao 2024 Dataset
# Complete migration pipeline with cleanup, validation, and retraining

param(
    [switch]$DryRun = $false,
    [switch]$SkipCleanup = $false,
    [switch]$SkipDownload = $false,
    [switch]$SkipTraining = $false,
    [switch]$Verbose = $false
)

# Configuration
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$ConfigFile = Join-Path $ProjectRoot "configs/train_rf_1M.yaml"

# Colors for output
$Success = "Green"
$Warning = "Yellow"
$Error = "Red"
$Info = "Cyan"

function Write-Section {
    param([string]$Title, [string]$Color = $Info)
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor $Color
    Write-Host $Title -ForegroundColor $Color
    Write-Host "=" * 80 -ForegroundColor $Color
}

function Write-Status {
    param([string]$Message, [string]$Status = "INFO", [string]$Color = $Info)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "$Timestamp | $Status | $Message" -ForegroundColor $Color
}

function Test-PythonEnvironment {
    Write-Section "Checking Python Environment" $Info
    
    # Check Python installation
    try {
        $PythonVersion = python --version 2>&1
        Write-Status "Python found: $PythonVersion" "OK" $Success
    }
    catch {
        Write-Status "Python not found. Please install Python 3.10+" "ERROR" $Error
        exit 1
    }
    
    # Check required packages
    $Packages = @("pandas", "numpy", "scikit-learn", "scapy", "pyarrow")
    foreach ($Package in $Packages) {
        try {
            python -c "import $Package" 2>&1 | Out-Null
            Write-Status "$Package: OK" "PACKAGE" $Success
        }
        catch {
            Write-Status "$Package: MISSING (install with pip install $Package)" "WARN" $Warning
        }
    }
}

function Invoke-CleanupPhase {
    param([switch]$Backup = $true)
    
    Write-Section "PHASE 1: Cleanup Old Data" $Warning
    
    if ($SkipCleanup) {
        Write-Status "Skipping cleanup phase (--SkipCleanup flag)" "SKIP" $Warning
        return
    }
    
    # Step 1: Dry-run first
    Write-Status "Running DRY-RUN to show what will be deleted..." "CLEANUP" $Info
    & python (Join-Path $ProjectRoot "scripts/utils/cleanup_old_data.py") --dry-run
    
    if ($DryRun) {
        Write-Status "Dry-run mode: No files deleted" "INFO" $Warning
        return
    }
    
    # Step 2: Get user confirmation
    $Confirmation = Read-Host "`nProceed with cleanup and move files to .trash? (yes/no)"
    if ($Confirmation -ne "yes") {
        Write-Status "Cleanup cancelled by user" "CANCEL" $Warning
        return
    }
    
    # Step 3: Execute cleanup
    Write-Status "Executing cleanup with backup..." "CLEANUP" $Warning
    & python (Join-Path $ProjectRoot "scripts/utils/cleanup_old_data.py") `
        --execute --backup --data-dir (Join-Path $ProjectRoot "data")
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Cleanup phase failed" "ERROR" $Error
        exit 1
    }
    
    Write-Status "✓ Cleanup phase complete" "OK" $Success
}

function Invoke-IntegrationPhase {
    Write-Section "PHASE 2: Download & Parse Gao 2024 Dataset" $Info
    
    if ($DryRun) {
        Write-Status "Dry-run mode: Skipping actual download" "DRY-RUN" $Warning
        return
    }
    
    $OutputDir = Join-Path $ProjectRoot "data/raw"
    $Args = @("--output", $OutputDir)
    
    if ($SkipDownload) {
        Write-Status "Using existing Gao 2024 source (--SkipDownload flag)" "INFO" $Warning
        $Args += "--skip-download"
    }
    
    Write-Status "Starting Gao 2024 integration..." "INTEGRATE" $Info
    Write-Status "Repository: https://github.com/ggyggy666/DNS-Tunnel-Datasets" "INFO" $Info
    
    & python (Join-Path $ProjectRoot "scripts/data/integrate_gao2024.py") @Args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Integration phase failed" "ERROR" $Error
        exit 1
    }
    
    Write-Status "✓ Integration phase complete" "OK" $Success
}

function Invoke-ValidationPhase {
    Write-Section "PHASE 3: Validate Gao 2024 Dataset" $Info
    
    if ($DryRun) {
        Write-Status "Dry-run mode: Skipping validation" "DRY-RUN" $Warning
        return
    }
    
    $InputFile = Join-Path $ProjectRoot "data/raw/gao_dns_tunnel_2024_parsed.parquet"
    $OutputFile = Join-Path $ProjectRoot "outputs/gao2024_validation_report.json"
    
    if (-Not (Test-Path $InputFile)) {
        Write-Status "Input file not found: $InputFile" "ERROR" $Error
        exit 1
    }
    
    Write-Status "Validating dataset quality..." "VALIDATE" $Info
    & python (Join-Path $ProjectRoot "scripts/data/validate_gao2024.py") `
        --input $InputFile `
        --output $OutputFile
    
    $ValidationExitCode = $LASTEXITCODE
    
    if ($ValidationExitCode -eq 0) {
        Write-Status "✓ Validation phase complete (all checks passed)" "OK" $Success
    }
    elseif ($ValidationExitCode -eq 2) {
        Write-Status "✓ Validation phase complete with warnings" "WARN" $Warning
        Write-Status "Review report at: $OutputFile" "INFO" $Info
    }
    else {
        Write-Status "Validation phase failed (critical issues found)" "ERROR" $Error
        Write-Status "Review report at: $OutputFile" "INFO" $Info
        exit 1
    }
}

function Invoke-PipelineRebuildPhase {
    Write-Section "PHASE 4: Rebuild Data Pipeline" $Info
    
    if ($DryRun) {
        Write-Status "Dry-run mode: Skipping pipeline rebuild" "DRY-RUN" $Warning
        return
    }
    
    $InputDir = Join-Path $ProjectRoot "data/raw"
    $OutputDir = Join-Path $ProjectRoot "data/splits_gao2024"
    
    Write-Status "Rebuilding pipeline with Gao 2024 data..." "PIPELINE" $Info
    Write-Status "Input: $InputDir" "INFO" $Info
    Write-Status "Output: $OutputDir" "INFO" $Info
    
    & python (Join-Path $ProjectRoot "scripts/data/build_pipeline.py") `
        --input-dir $InputDir `
        --output-dir $OutputDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Pipeline rebuild failed" "ERROR" $Error
        exit 1
    }
    
    Write-Status "✓ Pipeline rebuild complete" "OK" $Success
}

function Invoke-TrainingPhase {
    Write-Section "PHASE 5: Train Model on Gao 2024 Data" $Info
    
    if ($SkipTraining) {
        Write-Status "Skipping training phase (--SkipTraining flag)" "SKIP" $Warning
        return
    }
    
    if ($DryRun) {
        Write-Status "Dry-run mode: Skipping training" "DRY-RUN" $Warning
        return
    }
    
    Write-Status "Starting model training..." "TRAIN" $Info
    Write-Status "Config: $ConfigFile" "INFO" $Info
    
    & python (Join-Path $ProjectRoot "scripts/train/train_rf_1M.py") `
        --config $ConfigFile `
        --run-name "gao2024-production"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Training phase failed" "ERROR" $Error
        exit 1
    }
    
    Write-Status "✓ Training phase complete" "OK" $Success
}

function Invoke-EvaluationPhase {
    Write-Section "PHASE 6: Evaluate Model" $Info
    
    if ($DryRun) {
        Write-Status "Dry-run mode: Skipping evaluation" "DRY-RUN" $Warning
        return
    }
    
    Write-Status "Evaluating model on test data..." "EVAL" $Info
    
    # Check if evaluation script exists
    $EvalScript = Join-Path $ProjectRoot "scripts/evaluate/evaluate_on_real_data.py"
    if (Test-Path $EvalScript) {
        & python $EvalScript
        if ($LASTEXITCODE -ne 0) {
            Write-Status "Evaluation failed" "WARN" $Warning
        }
        else {
            Write-Status "✓ Evaluation complete" "OK" $Success
        }
    }
    else {
        Write-Status "Evaluation script not found: $EvalScript" "SKIP" $Warning
    }
}

function Show-Summary {
    Write-Section "MIGRATION COMPLETE" $Success
    
    Write-Status "✅ All phases completed successfully!" "SUCCESS" $Success
    Write-Status ""
    Write-Status "Dataset Location: $(Join-Path $ProjectRoot 'data/raw/gao_dns_tunnel_2024_parsed.parquet')" "INFO" $Info
    Write-Status "Validation Report: $(Join-Path $ProjectRoot 'outputs/gao2024_validation_report.json')" "INFO" $Info
    Write-Status "New Data Splits: $(Join-Path $ProjectRoot 'data/splits_gao2024')" "INFO" $Info
    Write-Status ""
    Write-Status "Next Steps:" "INFO" $Info
    Write-Status "  1. Review validation report" "INFO" $Info
    Write-Status "  2. Compare model performance metrics" "INFO" $Info
    Write-Status "  3. Update deployment configs" "INFO" $Info
    Write-Status ""
    Write-Status "Documentation:" "INFO" $Info
    Write-Status "  - Citation: Gao et al. (2024)" "INFO" $Info
    Write-Status "  - Repository: https://github.com/ggyggy666/DNS-Tunnel-Datasets" "INFO" $Info
}

# Main execution
function Main {
    Write-Section "DNS TUNNEL DETECTION - GAO 2024 MIGRATION WORKFLOW" $Success
    
    # Display configuration
    Write-Status "Dry-run mode: $DryRun" "CONFIG" $Info
    Write-Status "Skip cleanup: $SkipCleanup" "CONFIG" $Info
    Write-Status "Skip download: $SkipDownload" "CONFIG" $Info
    Write-Status "Skip training: $SkipTraining" "CONFIG" $Info
    Write-Status "Project root: $ProjectRoot" "CONFIG" $Info
    
    # Pre-flight checks
    Test-PythonEnvironment
    
    # Execute phases
    Invoke-CleanupPhase -Backup $true
    Invoke-IntegrationPhase
    Invoke-ValidationPhase
    Invoke-PipelineRebuildPhase
    Invoke-TrainingPhase
    Invoke-EvaluationPhase
    
    # Summary
    Show-Summary
}

# Run main
try {
    Main
    exit 0
}
catch {
    Write-Host ""
    Write-Section "WORKFLOW FAILED" $Error
    Write-Status $_.Exception.Message "ERROR" $Error
    Write-Status $_.ScriptStackTrace "DEBUG" $Error
    exit 1
}
