use std::ffi::CString;
use std::fs::File;
use std::io;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Default)]
pub struct BWX {
    data: Cursor<Vec<u8>>,
    size: usize,
    blocks: usize,
    pub pointer: usize,
}

impl BWX {
    pub fn new() -> Self {
        BWX { ..Default::default() }
    }

    pub fn load_from_file(&mut self, filename: &str) -> Result<(), String> {
        println!("Read from file: {filename}");

        let data = std::fs::read(filename).unwrap();
        self.data = Cursor::new(data);
        //println!("{:?}", data);

        if self.read_string(4) != "BWXF".to_string() {
            return Err("Not BWXF format!".to_string());
        }

        //let mut sz = 0;
        /*
        let sz = self.data.read_u32::<LittleEndian>().unwrap();
        let blocks = self.data.read_u32::<LittleEndian>().unwrap();

         */
        let sz = self.read_i32_packed();
        let blocks = self.read_i32_packed();
        println!("sz: {}, blocks: {}", sz, blocks);

        //eprintln!("Pointer: {}", self.pointer);
        eprintln!("Pointer: {}", self.data.position());

        Ok(())
    }

    fn read_string(&mut self, length: usize) -> String {
        let mut buffer = Vec::new();
        buffer.resize(length, 0);

        self.data.read(&mut buffer).unwrap();

        let c = String::from_utf8_lossy(&buffer).to_string();
        println!("{:?}", buffer);
        c
    }

    fn read_i32_packed(&mut self) -> i32 {
        let mut result: u32 = 0;
        let mut shift = 0;

        while shift < 35 {
            let t = self.data.read_u8().unwrap() as u32;
            println!("t: {}", t);
            result |= (t & 0x7f) << shift;
            println!("{}", result);
            if t & 0x80 == 0 {
                break;
            } else {
                shift += 7;
            }
        }

        result as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bwx_header() {
        let mut bwx = BWX::new();
        bwx.load_from_file("EXTERNAL_UI_DEFAULT.PNX");
        bwx.data.set_position(0);
        assert_eq!(bwx.read_string(4), "BWXF".to_string(), "File header is not BWXF!");
    }
}