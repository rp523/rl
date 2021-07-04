#coding: utf-8

class LogWriter:
    def __init__(self, dst_path):
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents = True)
        self.__fp = open(dst_path, "w")
        self.__first = True

    def write(self, out_infos):
        if self.__first:
            LogWriter.__write(self.__fp, True, out_infos)
            self.__first = False
        LogWriter.__write(self.__fp, False, out_infos)

    @staticmethod
    def __write(fp, header, out_infos):
        assert(not (fp is None))
        for i, (key, val) in enumerate(out_infos.items()):
            if header:
                out = key
            else:
                out = val
            fp.write("{},".format(out))
        fp.write("\n")