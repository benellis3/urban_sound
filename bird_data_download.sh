function substitute_column () 
{

    if find $3 -name "*.txt" -type f -exec sed -i.bak -e "s/$1/$2/" {} \; 
    then
        # clean up the backup files
        find $3 -name "*.bak" -type f -delete
    else
        echo "Find exited with an error"
        exit 1
    fi
}

mkdir -p data/bird_data
pushd data/bird_data


# Download the dataset
wget -O bird_data.zip https://esajournals.onlinelibrary.wiley.com/action/downloadSupplement\?doi\=10.1002%2Fecy.3329\&file\=ecy3329-sup-0001-DataS1.zip
# unzip it
unzip bird_data.zip 

# For reasons unknown to science, the spacing in the annotation files is unbelievably inconsistent
# This means that pandas read_csv has a hard time parsing it. Here we use sed to fix the column names
# so that it is easier to parse.

substitute_column "Begin Time (s)" "begin_time" .
substitute_column "End Time (s)" "end_time" .
substitute_column "Low Freq (Hz)" "low_freq" .
substitute_column "High Freq (Hz)" "high_freq" .
popd
