# corduroy

```
                              .                   .                     .   .             
      .   .     .  .   ..                     .%             .                             
                  ..             . .         .-@:          .                  .   .        
  :+    .         .          .              .       =++=- .  .---==---.           .  .@.   
:*@@*. .                .     .         .::----:. -::-+**#%%%####**##*++++*#=.--++--.     .
  =*        .                   .-=+#%%%#*##****++%+--.=+++++++++++++++*++++=:--+**#%###=: .
   .                .     .  -*%%#++++++++=---=++++*=- -+*++++=++++++=+++++-=+----=+=+++*++=
    .   . *@.     .       -#%#*++++++++++++++++=----+----+-**+++++++*+++*+++++++=---++++++++
.      .   .  .         -***++++----------==+++++++--+***+-=*+*++-*+**++++++++*+++---*++++++
              .    .   -++++---.:::+***- .  ===--+++-+##*+-*++*++++++++++*++++++----=+=++==
     .                =*+=-: :+:=:***+#######-  ---+-=###*+++++***+*++++*+**++++---- -=-----
 .  .  .      ..     =++-:=++-:+##***#####**++**. .:-**##*=**++++***++++**++=+++----.-------
                    -+- =***##-*%#=#######*=-*****  +++#*==#++++*+=++*+++*++++=---- --- ----
                    +- *********+####### -*********:  ===-%*+++++***++++++++=+----        :
                    = +*******:#########***+=*==****- :---**+==*+++++*++++=------.    .    
                  .  :******=###########=**-+####****   :- =++--- ===+++-------           .
    .     .=@-.     .#=**+=#############*********=**=.   -=**++++--:---------      .     . 
       .    *       *#**###*###########***********=**#  ===#**+++---------..     .         
              .     ####=*#############**************#  ==%#+**++-----.       ..      .    
      =.            +#####=****#########************#=+==##++=+----   . .      .  :        
       .           . #####******+############******##--*%#*+++---               .+@. .     
 . .               ..=####**********#########******:=+*%**++---     .         ..     .     
                   .  +#####*******##########****:--+#**++---          . ..                
          .           .=#####****############***:-+**++----                   .            
-.                .      ####**=#############- -+*+----- .          .      .            .  
@@:                 .      ##**#############=------          .                      .+    
    .   .      +.            =###########+.   .          .                       .   --
        .             .  . .  .                                                      . :.  
  .   .           .   .                                              .          .
```

Adaptive mesh precipitation mapping from ERA5 Zarr v3 on Google Cloud Storage.

## Prerequisites

- Rust >= 1.91 (`rustup update` if needed)
- Internet connection (reads from public GCS bucket, no credentials required)

## Build

```bash
cargo build --release
```

## Usage

Fetch ERA5 total precipitation for a bounding box at a specific UTC datetime:

```bash
cargo run --release -- \
  --bbox "lat_min,lat_max,lon_min,lon_max" \
  --datetime "YYYY-MM-DDTHH:MM:SS" \
  -o output.png
```

### Examples

Central US:
```bash
cargo run --release -- --bbox "35.0,45.0,-90.0,-75.0" --datetime "2023-06-15T12:00:00" -o precip.png
```

Western Europe:
```bash
cargo run --release -- --bbox "42.0,52.0,-5.0,10.0" --datetime "2023-01-10T06:00:00" -o europe.png
```

Southeast Asia:
```bash
cargo run --release -- --bbox "-10.0,20.0,95.0,120.0" --datetime "2023-08-01T00:00:00" -o asia.png
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--bbox` | Bounding box as `lat_min,lat_max,lon_min,lon_max`. Latitude: [-90, 90]. Longitude: [-180, 360]. |
| `--datetime` | UTC datetime in ISO 8601 format. Data available from 1940 to ~2023. |
| `-o, --output` | Output PNG path (default: `precip.png`). |

### Verbose logging

```bash
RUST_LOG=debug cargo run --release -- --bbox "35.0,45.0,-90.0,-75.0" --datetime "2023-06-15T12:00:00"
```

## Testing

Run offline tests (domain validation + plot rendering):
```bash
cargo test
```

Run integration test (fetches real data from GCS):
```bash
cargo test -- --ignored
```

Run all tests:
```bash
cargo test -- --include-ignored
```

## Data Source

Reads from the [ARCO ERA5](https://github.com/google-research/arco-era5) Zarr v3 store on Google Cloud Storage:

- **Bucket**: `gs://gcp-public-data-arco-era5`
- **Variable**: `total_precipitation` (meters of water, hourly)
- **Resolution**: 0.25 degrees (~25 km), hourly
- **Coverage**: Global, 1940-2023
- **Access**: Anonymous (no credentials needed)

Output values are displayed in millimeters (mm).
