process decompress_scalefactors {
    input:
    path scalefactors_json_gz from scalefactors_json_gz_ch

    output:
    path "*.json" into scalefactors_json_ch

    script:
    """
    gunzip -c $scalefactors_json_gz > ${scalefactors_json_gz.baseName}.json
    """
}

process load_json_files {
    input:
    path json_files from scalefactors_json_ch

    output:
    path "*.json" into loaded_json_ch

    script:
    """
    echo "Loading JSON files: $json_files"
    """
}

process extract_scalefactors {
    input:
    path json_file from loaded_json_ch

    output:
    path "scalefactors.txt" into scalefactors_txt_ch

    script:
    """
    jq '.scalefactors' $json_file > scalefactors.txt
    """
}

process compare_scalefactors {
    input:
    path scalefactors_txt from scalefactors_txt_ch

    script:
    """
    echo "Comparing scalefactors between CLM 01 and CLM 02 samples"
    diff -y --suppress-common-lines CLM_01_scalefactors.txt CLM_02_scalefactors.txt
    """
}

process confirm_resolution {
    input:
    path hires_images_gz from hires_images_gz_ch

    output:
    path "*.png" into confirmed_resolution_ch

    script:
    """
    gunzip -c $hires_images_gz > ${hires_images_gz.baseName}.png
    echo "High-resolution image decompressed: ${hires_images_gz.baseName}.png"
    """
}