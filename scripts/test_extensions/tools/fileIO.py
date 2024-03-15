import asyncio
from pydantic import Field
from schema_agents import schema_tool, Role

@schema_tool
async def write_to_file(string : str = Field(description="The string to write to a file"),
                        out_file : str = Field(description="The file to write to")) -> str:
    """Write a string to a file. Use this when output are too large to write to stdout or as a return string to the user"""
    with open(out_file, 'w') as f:
        f.write(string)
    return "Success"

@schema_tool
async def read_file(string : str = Field(description="The file to read from")) -> str:
    """Read a file and return the string"""
    with open(string, 'r') as f:
        return f.read()
    
@schema_tool
async def ftp_download(ftp_host : str = Field(description="The FTP host to download from. E.g. for ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/00/00/PMC1790863.tar.gz, the host is ftp.ncbi.nlm.nih.gov"),
                       remote_file_path : str = Field(description="The file path to download from the FTP server. E.g. for ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/00/00/PMC1790863.tar.gz, the file path is pub/pmc/oa_package/00/00/PMC1790863.tar.gz"),
                       out_file : str = Field(description="The file to write to. E.g. for ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/00/00/PMC1790863.tar.gz the out_file is PMC1790863.tar.gz")) -> str:
    """Download a file from an FTP server"""
    import ftplib
    with open(out_file, 'wb') as f:
        ftp = ftplib.FTP(ftp_host)
        ftp.login()  # Might need credentials
        ftp.retrbinary('RETR ' + remote_file_path, f.write)
        ftp.quit()
    return f"Success - file downloaded from {ftp_host} to {out_file}"

@schema_tool
async def unzip_tar_gz(file : str = Field(description="The file to unzip")) -> str:
    """Unzip a .tar.gz file"""
    import tarfile
    with tarfile.open(file, "r:gz") as tar:
        tar.extractall()
    return f"Success - file unzipped to {file.replace('.tar.gz', '')}"

@schema_tool
async def list_files_in_dir(dir : str = Field(description="The directory to list files in")) -> list[str]:
    """List all files in a directory"""
    import os
    return os.listdir(dir)
    

async def test_ftp_download():
    url = "ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/00/00/PMC1790863.tar.gz"
    out_file = "PMC1790863.tar.gz"
    ftp_host = "ftp.ncbi.nlm.nih.gov"
    remote_file_path = "pub/pmc/oa_package/00/00/PMC1790863.tar.gz"
    res = await ftp_download(ftp_host = ftp_host,
                                remote_file_path = remote_file_path,
                             out_file = out_file)
    print(res)

async def test_unzip_tar_gz():
    file = "PMC1790863.tar.gz"
    res = await unzip_tar_gz(file = file)
    print(res)

async def main():
    await test_ftp_download()
    await test_unzip_tar_gz()
    res = await list_files_in_dir(dir = "./PMC1790863/")
    print(res)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
    
