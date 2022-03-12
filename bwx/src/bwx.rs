use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};
use tracing::{debug, error, info, warn};

// u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
enum SlData {
    U8(u8),
    I16(i16),
    I32(i32),
    F32(f32),
    String(String),
    BlockSizeNumber(i32, i32),
    None,
}

#[derive(Debug, Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    content: Cursor<Vec<u8>>,
    size: i32,
    blocks: i32,
    pub pointer: usize,

}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

static mut COUNT: u32 = 0;

impl BWX {
    /// Returns a BWX with the given file name
    ///
    // /// TODO: Merge load_from_file() to new()
    // /// # Arguments
    // ///
    // /// * `path` - A string slice that holds the file name of a BNX / PNX file
    // ///
    /// # Examples
    ///
    /// ```
    /// use bwx::BWX;
    /// let bwx = BWX::new();
    /// ```
    pub fn new() -> Self {
        BWX { ..Default::default() }
    }

    #[tracing::instrument(skip(self, filename))]
    pub fn load_from_file(&mut self, filename: &str) -> Result<()> {
        debug!(filename);

        let data = std::fs::read(filename)?;
        self.content = Cursor::new(data);

        self.check_bwx_header()?;
        self.content.set_position(4);

        (self.size, self.blocks) = self.read_block_size_number()?;
        debug!(self.size, self.blocks);

        for i in 0..self.blocks {
            debug!("Block {i}");
            // TODO: Debug, process no more than 3 blocks
            if i == 3 {
                break;
            }

            let name = self.read_string()?;
            debug!("section name: {name}");
            let _data = self.parse_block();
            /*
            match name.as_str() {
                "0" => {
                    // TODO: Add the data to Vector
                    let _data = self.parse_block();
                }
                "HEAD" => {
                    self.parse_head()?;
                }
                "MTRL" => {
                    self.parse_material()?;
                }
                _ => {
                    error!("Unhandled section: {name}");
                    panic!("Unhandled section: {name}");
                }
            }

             */
        }

        Ok(())
    }

    /// Check whether the file is a valid BNX / PNX format
    fn check_bwx_header(&mut self) -> Result<()> {
        let header = &self.content.get_ref()[..4];
        if header != "BWXF".as_bytes() {
            return Err("Invalid BWX file.")?;
        }

        Ok(())
    }

    /// Read BNX / PNX special packed integer value (little endian)
    fn read_i32_packed(&mut self) -> Result<i32> {
        let mut result: u32 = 0;
        let mut shift = 0;

        while shift < 35 {
            let t = self.content.read_u8()? as u32;
            result |= (t & 0x7f) << shift;

            if t & 0x80 == 0 {
                break;
            } else {
                shift += 7;
            }
        }

        Ok(result as i32)
    }

    /// Read block size & numbers
    fn read_block_size_number(&mut self) -> Result<(i32, i32)> {
        Ok((self.read_i32_packed()?, self.read_i32_packed()?))
    }


    /// Read string
    fn read_string(&mut self) -> Result<String> {
        let length = self.content.read_u8()?;
        let mut buffer = Vec::new();
        buffer.resize(length as usize, 0);
        self.content.read_exact(&mut buffer)?;

        let (cow, _encoding, had_errors) = encoding_rs::EUC_KR.decode(&buffer);
        if had_errors {
            error!("Failed to convert string from Korean to UTF-8!");
            Ok(String::from_utf8_lossy(&buffer).trim_matches('\0').to_string())
        } else {
            Ok(cow.to_string())
        }
    }

    ///
    #[tracing::instrument(skip(self))]
    fn parse_block(&mut self) -> Result<SlData> {
        let sl_type1 = self.content.read_u8()?;
        let sl_type = char::from(sl_type1);

        debug!("SL_TYPE: {sl_type}, {:04x}", sl_type1);
        unsafe {
            COUNT += 1;
            if COUNT > 50 {
                panic!("Over 100 times");
            }
        }

        match sl_type {
            'S' => {
                let value = self.read_string()?;
                debug!("String: {value}");
                Ok(SlData::String(value))
            }
            'A' => {
                let (size, blocks) = self.read_block_size_number()?;
                debug!("Block size and numbers: {size}, {blocks}");

                for i in 0..blocks {
                    debug!("---------- Block {}", i+1);
                    let _data = self.parse_block();
                    //blocks -= 1;
                }
                Ok(SlData::None)
                //Ok(SlData::BlockSizeNumber(size, blocks))
            }
            'I' => {
                let value = self.content.read_i32::<LittleEndian>()?;
                debug!("Integer: {value}");
                Ok(SlData::I32(value))
            }
            'W' => {
                let value = self.content.read_i16::<LittleEndian>()?;
                debug!("Word: {value}");
                Ok(SlData::I16(value))
            }
            'F' => {
                let value = self.content.read_f32::<LittleEndian>()?;
                debug!("Float: {:.3}", value);
                Ok(SlData::F32(value))
            }
            t  if (t as u32) < 0x20 => {
                // TODO: Confirm what's going on and return proper value
                warn!("UNKNOWN TYPE, treat as one unsigned byte, 0x{:02x}", t as u32);
                Ok(SlData::U8(t as u8))
            }
            _ => {
                error!("Unhandled type {sl_type}, pointer: {}", self.content.position());
                panic!("Unhandled type {sl_type}");
            }
        }
    }

    /// Parse 'HEAD" block
    #[tracing::instrument(skip(self))]
    fn parse_head(&mut self) -> Result<()> {
        let (size, mut blocks) = match_block_size_number(self.parse_block()?)?;
        debug!("HEAD: size = {size}, blocks = {blocks}");
        while blocks > 0 {
            let _data = self.parse_block();
            blocks -= 1;
        }

        Ok(())
    }

    /// Parse 'MTRL" block
    #[tracing::instrument(skip(self))]
    fn parse_material(&mut self) -> Result<()> {
        let (size, mut blocks) = match_block_size_number(self.parse_block()?)?;
        debug!("MTRL: size = {size}, blocks = {blocks}");
        while blocks > 0 {
            let _data = self.parse_block();
            blocks -= 1;
        }

        Ok(())
    }
}

/// Match block size & numbers
fn match_block_size_number(data: SlData) -> Result<(i32, i32)> {
    match data {
        SlData::BlockSizeNumber(s, b) => Ok((s, b)),
        _ => Err("Cannot match SlData::BlockSizeNumber")?,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_bwx_header() {
        let mut bwx = BWX { ..Default::default() };
        bwx.content = Cursor::new(vec![66, 87, 88, 70]);
        assert!(bwx.check_bwx_header().is_ok(), "File header check should pass");
        bwx.content = Cursor::new(vec![11, 22, 33, 44]);
        assert!(bwx.check_bwx_header().is_err(), "File header check should fail");
    }

    #[test]
    fn read_i32_packed() {
        let mut bwx = BWX { ..Default::default() };
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.content = Cursor::new(vec![0xc1, 0xef, 0x5a, 0x0c]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 1488833, "Packed integer value incorrect");
        bwx.content = Cursor::new(vec![0x0c, 0x02]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 12, "Packed integer value incorrect");
    }

    #[test]
    fn read_string() {
        let mut bwx = BWX { ..Default::default() };
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.content = Cursor::new(vec![0x02, 0x30, 0x00, 0x53]);
        assert_eq!(bwx.read_string().unwrap().as_str(), "0", "The string should be '0'");
    }
}

