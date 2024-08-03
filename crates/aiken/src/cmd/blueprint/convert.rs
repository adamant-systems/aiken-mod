use aiken_project::{
    blueprint::{error::Error as BlueprintError, Blueprint},
    config::Config,
    error::Error as ProjectError,
};
use clap::ValueEnum;
use miette::IntoDiagnostic;
use serde_json::json;
use std::{env, fs::File, io::BufReader, path::PathBuf, process};

/// Convert a blueprint into other formats.
#[derive(clap::Args)]
pub struct Args {
    /// Path to project
    directory: Option<PathBuf>,

    /// Name of the validator's module within the project. Optional if there's only one validator.
    #[clap(short, long)]
    module: Option<String>,

    /// Name of the validator within the module. Optional if there's only one validator.
    #[clap(short, long)]
    validator: Option<String>,

    // Format to convert to
    #[clap(long, default_value = "cardano-cli")]
    to: Format,
}

#[derive(Copy, Clone, ValueEnum)]
pub enum Format {
    CardanoCli,
}

pub fn exec(
    Args {
        directory,
        module,
        validator,
        to,
    }: Args,
) -> miette::Result<()> {
    let title = module.as_ref().map(|m| {
        format!(
            "{m}{}",
            validator
                .as_ref()
                .map(|v| format!(".{v}"))
                .unwrap_or_default()
        )
    });

    let title = title.as_ref().or(validator.as_ref());

    let project_path = if let Some(d) = directory {
        d
    } else {
        env::current_dir().into_diagnostic()?
    };

    let blueprint_path = project_path.join("plutus.json");

    // Read blueprint
    let blueprint = File::open(blueprint_path)
        .map_err(|_| BlueprintError::InvalidOrMissingFile)
        .into_diagnostic()?;

    let blueprint: Blueprint =
        serde_json::from_reader(BufReader::new(blueprint)).into_diagnostic()?;

    let opt_config = Config::load(&project_path).ok();

    let cardano_cli_type = opt_config
        .map(|config| config.plutus)
        .unwrap_or_default()
        .cardano_cli_type();

    // Perform the conversion
    let when_too_many =
        |known_validators| ProjectError::MoreThanOneValidatorFound { known_validators };
    let when_missing = |known_validators| ProjectError::NoValidatorNotFound { known_validators };

    let result =
        blueprint.with_validator(title, when_too_many, when_missing, |validator| match to {
            Format::CardanoCli => {
                let cbor_bytes = validator.program.to_cbor().unwrap();

                let mut double_cbor_bytes = Vec::new();

                let mut cbor_encoder = pallas_codec::minicbor::Encoder::new(&mut double_cbor_bytes);

                cbor_encoder.bytes(&cbor_bytes).unwrap();

                let cbor_hex = hex::encode(double_cbor_bytes);

                Ok(json!({
                    "type": cardano_cli_type,
                    "description": "Generated by Aiken",
                    "cborHex": cbor_hex
                }))
            }
        });

    match result {
        Ok(value) => {
            let json = serde_json::to_string_pretty(&value).unwrap();

            println!("{json}");

            Ok(())
        }
        Err(err) => {
            err.report();

            process::exit(1)
        }
    }
}
