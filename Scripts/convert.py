import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: convert.py infile outfile')
        sys.exit(0)
    ip_filename = sys.argv[1]
    with open(ip_filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    op_filename = sys.argv[2]
    with open(op_filename, 'w+') as out:
        for l in lines:
            splitted = l.split("\t")
            to_write = []
            for i in range(1,len(splitted)):
                to_write.append(splitted[i])
            sent = "\t".join(to_write)
            out.write(sent + "\n")
    out.close()


