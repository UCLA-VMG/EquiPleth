import configparser
import os
import sys

config = configparser.ConfigParser()
#find absolute path to this directory
this_dir = os.path.dirname(os.path.abspath(__file__))
#add to PYTHONPATH
sys.path.insert(0, this_dir)
#Read in config to make avaliable to all other libraries
config.read(os.path.join(this_dir, "configs", "sensors_config.ini"))
#add absolute path as a config
config.set("mmhealth", "abs_path", this_dir)

if __name__ == '__main__':

    for section in config.sections():
        print(section)
        section_dict = dict(config[section])
        for key in section_dict:
            print("  {} = {}".format(key,section_dict[key]))
        print()

    print(config.getint('rgb', 'fps'))
    print(config.get("mmhealth", "data_path").encode('unicode_escape'))
    print(os.listdir(config.get("mmhealth", "data_path").encode('unicode_escape')))
    