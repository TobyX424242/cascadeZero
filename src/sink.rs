use crate::error::DataCollectionError;
use crate::model::TrainingSample;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub trait DatasetSink {
    fn write_sample(&mut self, sample: &TrainingSample) -> Result<(), DataCollectionError>;
    fn flush(&mut self) -> Result<(), DataCollectionError>;
}

pub struct JsonlDatasetSink<W: Write> {
    writer: BufWriter<W>,
}

impl<W: Write> JsonlDatasetSink<W> {
    pub fn new(inner: W) -> Self {
        Self {
            writer: BufWriter::new(inner),
        }
    }

    pub fn into_inner(mut self) -> Result<W, DataCollectionError> {
        self.writer.flush()?;
        self.writer
            .into_inner()
            .map_err(|error| DataCollectionError::Io(error.into_error()))
    }
}

impl JsonlDatasetSink<File> {
    pub fn create<P>(path: P) -> Result<Self, DataCollectionError>
    where
        P: AsRef<Path>,
    {
        Ok(Self::new(File::create(path)?))
    }
}

impl<W: Write> DatasetSink for JsonlDatasetSink<W> {
    fn write_sample(&mut self, sample: &TrainingSample) -> Result<(), DataCollectionError> {
        serde_json::to_writer(&mut self.writer, sample)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), DataCollectionError> {
        self.writer.flush()?;
        Ok(())
    }
}
