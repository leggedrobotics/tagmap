#!/usr/bin/env bash

wget https://huggingface.co/datasets/frozendonuts/tag-mapping/resolve/main/demo_data.zip
echo "Unzipping demo data"
unzip -q demo_data.zip
rm demo_data.zip
echo "Done downloading and unzipping demo data"
