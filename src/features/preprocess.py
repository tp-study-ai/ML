import os


def clean_code(code: str) -> tuple[str, int, int, int]:
    pid = os.getpid()
    with open(f"/dev/shm/input_{pid}.cpp", 'w') as program:
        program.write(code.replace("\r\n", "\n").replace("\r", "\n"))

    os.system('clang-format -style="{IndentWidth: 4, ColumnLimit: 5000}" -i ' + f'/dev/shm/input_{pid}.cpp')

    input_file = f"/dev/shm/input_{pid}.cpp"
    output_file = f"/dev/shm/preprocess_{pid}.cpp"

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if "#ifdef" in line or "#ifndef" in line or "#endif" in line or "#else" in line or "#elif" in line:
                f_out.write(line)
                continue
            if "#pragma" in line:
                continue
            if "#include" in line or "#import" in line:
                continue
            else:
                f_out.write(line)

    os.system(f"g++ -E -w /dev/shm/preprocess_{pid}.cpp > /dev/shm/first_prep_{pid}.cpp 2>/dev/shm/err1_{pid}")

    os.system(f"sed -i '/# /d' /dev/shm/first_prep_{pid}.cpp")

    # os.system("sed -i '1s/^/#include <bits\/stdc++.h>\n/' /dev/shm/output_prep.cpp")

    #     preambule = ""

    with open(f"/dev/shm/err1_{pid}", 'r') as warn:
        if len(warn.read()) > 3:
            #             preambule = "//!\n"
            with open(output_file, "r") as res:
                ret = res.read()
                tdc = ret.count("typedef")
                uc = ret.count("using")
                return ret, tdc, uc, 2

    os.system('clang-format -style="{IndentWidth: 4, ColumnLimit: 5000}" -i ' + f'/dev/shm/first_prep_{pid}.cpp')

    with open(f'/dev/shm/first_prep_{pid}.cpp', "r") as f_in, open(f'/dev/shm/for_prep2_{pid}.cpp', "w") as f_out:
        for line in f_in:
            if line.startswith("type"):
                if "[" in line and "];" in line:
                    f_out.write(line)
                    continue
                alias = re.sub("[*&]", "", line.split()[-1])
                define = f"#define {alias[:-1]} {line.replace('typedef ', '').replace(alias, '')}"  # need rework
                f_out.write(define)
            else:
                f_out.write(line)

    os.system(f"g++ -E -w /dev/shm/for_prep2_{pid}.cpp > /dev/shm/second_prep_{pid}.cpp 2>/dev/shm/err2_{pid}")

    os.system(f"sed -i '/# /d' /dev/shm/second_prep_{pid}.cpp")

    with open(f"/dev/shm/err2_{pid}", 'r') as warn:
        if len(warn.read()) > 3:
            #             preambule = "//!\n"
            with open(f"/dev/shm/for_prep2_{pid}.cpp", "r") as res:
                ret = res.read()
                tdc = ret.count("typedef")
                uc = ret.count("using")
                return ret, tdc, uc, 1

    with open(f"/dev/shm/second_prep_{pid}.cpp", 'r') as res:
        ret = res.read()
        tdc = ret.count("typedef")
        uc = ret.count("using")
        return ret, tdc, uc, 0