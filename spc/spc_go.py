import argparse
import sys
from spc.spcserver import SPCServer

def parse_cmds():
    parser = argparse.ArgumentParser(description='Accessing spc.ucsd.edu pipeline')
    parser.add_argument('--search-param-file', default=None, help='spc.ucsd.edu search param path')
    parser.add_argument('--image-output-path', default=None, help='Downloaded images output path')
    parser.add_argument('--meta-output-path', default=None, help='Meta data output path')
    parser.add_argument('-d', '--download', default=False, help='Download flagging option')
    args = parser.parse_args(sys.argv[1:])
    return args



def validate_arguments(args):
    def fatal_error(msg):
        """ Prints out error message

        Function not in heavy usage
        #TODO make decisions to scrap or not

        Args:
            msg (str): Message content

        Returns:
            bool: Flag to indicate catch of fatal error

        """
        sys.stderr.write('%s\n' % msg)
        return True

    if (args.search_param_file is None):
        fatal_error("No search param file provided")
    if (args.search_param_file is not None) and ((args.meta_output_path is None) or (args.image_output_path is None)):
        fatal_error("No meta/image output path provided")
    if (args.search_param_file is not None) and not args.download and (args.image_output_path is not None):
        fatal_error("Download option not flagged")
    if (args.image_output_path is None) and args.download:
        fatal_error('No output image path provided.')



def main(args):
    print("Downloading images...")
    spc = SPCServer()
    spc.retrieve (textfile=args.search_param_file,
                  output_dir=args.image_output_path,
                  output_csv_filename=args.meta_output_path,
                  download=args.download)



if __name__ == '__main__':
    main(parse_cmds())



