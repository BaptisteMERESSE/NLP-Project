# Arkindex Archelec

## How to access the transcriptions ?

The transcriptions have been extracted form Arkindex using the script extract_text.py and are stored in the directory text_files in zip files. 

## How to extract the transcriptions from Arkindex

* Install the arkindex client
```
 virtualenv -p python3.11 venv
 source venv/bin/activate
 pip install arkindex-export
 ````


* Download the Arkindex archive

- go to https://demo.arkindex.org/browse/1bc39ca6-399b-47ca-9de1-ab2ef481cabb?top_level=true&folder=true
- menu Import/Export -> Manage database exports
- download the latest archive


* Run the extraction script

````
python extract_text.py
```
