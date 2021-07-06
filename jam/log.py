#coding: utf-8

class LogWriter:
    def __init__(self, dst_path):
        self.__dst_path = dst_path
        self.__first = True

    def write(self, out_infos):
        if self.__first:
            if not self.__dst_path.parent.exists():
                self.__dst_path.parent.mkdir(parents = True)
            with open(self.__dst_path, "w") as f:
                LogWriter.__write(f, True, out_infos)
            self.__first = False
        with open(self.__dst_path, "a") as f:
            LogWriter.__write(f, False, out_infos)

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