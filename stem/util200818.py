# coding=latin-1
import os
import sys
import re
import time
import pickle
import json
import numpy as np
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError


###############################################################################
def stringtime(n):
    h = str(int(n / 3600))
    m = str(int((n % 3600) / 60))
    s = str(int((n % 3600) % 60))
    if len(h) == 1: h = '0' + h
    if len(m) == 1: m = '0' + m
    if len(s) == 1: s = '0' + s
    return h + ':' + m + ':' + s


def steptime(start):
    now = time.time()
    dur = stringtime(now - start)
    print("time elapsed:", dur)
    return now


def start(sep=True):
    start = time.time()
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep: print('#'*80)
    print('start:', now)
    return start


def begin():
    start = time.time()
    return start


def end(start, sep=True):
    end = time.time()
    dur = end - start
    str_dur = stringtime(end - start)
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep:
        print('#'*80 + "\nend:", now, " - time elapsed:", str_dur + "\n" + '#'*80)
    else:
        print("end:", now, " - time elapsed:", str_dur)
    return dur


def now():
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    return now


def timefrom(start):
    now = time.time()
    return stringtime(now - start)


def printstep(string, index, step):
    n = index + 1
    if n % step == 0:
        print(string, 'step', n)
    return 1


###############################################################################
def say(*anything):
    printstr = ''
    for elem in list(anything): printstr += str(elem) + ' '
    printstr = re.sub(' $', '', printstr)
    sys.stdout.write("\r" + printstr)
    sys.stdout.flush()


###############################################################################
def file2str_withtail(pathfile, coding='utf-8'):
    '''
    input: (path to) file
    output: file as string
    '''
    with open(pathfile, 'r', encoding=coding) as input_file:
        out = input_file.read()
        # out = re.sub("\n+$", '', out) # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    return out


def file2str_withouttail(pathfile, coding='utf-8'):
    '''
    input: (path to) file
    output: file as string
    '''
    with open(pathfile, 'r', encoding=coding) as input_file:
        out = input_file.read()
        out = re.sub("\n+$", '', out) # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    return out


def tsv2npmatrix(pathfile, emptyend, sep="\t", filenc='utf-8', elemtype=str):
    '''
    input: tsv file
    output: list of lists 
    '''
    if emptyend: str_file = file2str_withtail(pathfile, coding=filenc)
    else:        str_file = file2str_withouttail(pathfile, coding=filenc)
    if   elemtype == int:   out = np.array([[int(elem)   for elem in row.split(sep)] for row in str_file.split("\n")])
    elif elemtype == float: out = np.array([[float(elem) for elem in row.split(sep)] for row in str_file.split("\n")])
    else:                   out = np.array([[str(elem)   for elem in row.split(sep)] for row in str_file.split("\n")])
    return out


def tsv2matrix(pathfile, emptyend, sep="\t", filenc='utf-8', elemtype=str):
    '''
    input: tsv file
    output: list of lists
    '''
    if emptyend: str_file = file2str_withtail(pathfile, coding=filenc)
    else:        str_file = file2str_withouttail(pathfile, coding=filenc)
    if   elemtype == int:   out = [[int(elem)   for elem in row.split(sep)] for row in str_file.split("\n")]
    elif elemtype == float: out = [[float(elem) for elem in row.split(sep)] for row in str_file.split("\n")]
    else:                   out = [[str(elem)   for elem in row.split(sep)] for row in str_file.split("\n")]
    return out


def file2list(pathfile, emptyend=True, filenc='utf-8', elemtype='str', sep=' '):
    if emptyend: fileraw = file2str_withtail(pathfile, coding=filenc)
    else:        fileraw = file2str_withouttail(pathfile, coding=filenc)
    if elemtype == 'int':
        out = [float(x) for x in fileraw.split(sep)] # se il format è float, non lo traduce direttamente
        out = [int(x) for x in out]
    elif elemtype == 'float':
        out = [float(x) for x in fileraw.split(sep)]
    else:
        out = [x for x in fileraw.split(sep)]
    return out


def file2dict(filename, keytype='str', valtype='str'):
    fileraw = file2str_withouttail(filename)
    #print(fileraw.split()) # splitta per spazi, tab, non vede la punteggiatura... insomma fa casino
    fileraw = re.sub("\n+$", '', fileraw)  # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    out = {}
    rows = fileraw.split("\n")
    for row in rows:
        cols = row.split("\t")
        if keytype == 'str' and valtype == 'int':
            out[cols[0]] = int(cols[1])
        elif keytype == 'str' and valtype == 'float':
            out[cols[0]] = float(cols[1])
        elif keytype == 'int' and valtype == 'int':
            out[int(cols[0])] = int(cols[1])
        elif keytype == 'float' and valtype == 'float':
            out[float(cols[0])] = float(cols[1])
        else:
            out[cols[0]] = cols[1]
    #for k in out: print(k, out[k])
    return out


def file2dictlist(filename, keytype='int', listype='int'):
    fileraw = file2str_withouttail(filename)
    #print(fileraw.split()) # splitta per spazi, tab, non vede la punteggiatura... insomma fa casino
    fileraw = re.sub("\n+$", '', fileraw)  # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    out = {}
    rows = fileraw.split("\n")
    for row in rows:
        cols = row.split("\t")
        if keytype == 'int' and listype == 'int':
            out[int(cols[0])] = [int(i) for i in cols[1:]]
        elif keytype == 'float' and listype == 'float':
            out[float(cols[0])] = [float(i) for i in cols[1:]]
        elif keytype == 'str' and listype == 'float':
            out[cols[0]] = [float(i) for i in cols[1:]]
        else:
            out[cols[0]] = [i for i in cols[1:]]
    #for k in out: print(k, out[k])
    return out


def file2dictset(filename, keytype='int', setype='int'):
    fileraw = file2str_withouttail(filename)
    #print(fileraw.split()) # splitta per spazi, tab, non vede la punteggiatura... insomma fa casino
    fileraw = re.sub("\n+$", '', fileraw)  # via il o gli ultimi \n, se ci sono righe vuote alla fine del file
    out = {}
    rows = fileraw.split("\n")
    for row in rows:
        cols = row.split("\t")
        if keytype == 'int' and setype == 'int':
            out[int(cols[0])] = {int(i) for i in cols[1:]}
        elif keytype == 'float' and setype == 'float':
            out[float(cols[0])] = {float(i) for i in cols[1:]}
        else:
            out[cols[0]] = {i for i in cols[1:]}
    #for k in out: print(k, out[k])
    return out


###############################################################################
def list2file(lis, fileout, sepline="\n", wra='w'):
    with open(fileout, wra) as fileout: [fileout.write(str(x) + sepline) for x in lis]
    return 1


def tuple2file(tup, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for item in tup: f_out.write(str(item[0]) + "\t" + str(item[1]) + "\n")
    return 1


def dict2file(dic, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for k in dic: f_out.write(str(k) + "\t" + str(dic[k]) + "\n")
    return 1


def docs4words2tsv(matrix, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        # f_out.write("\n".join(["\t".join(row) for row in matrix])) # scrivo invece riga per riga, o i file troppo grandi saltano
        for row in matrix[:-1]:
            f_out.write("\t".join(row) + "\n")
        f_out.write("\t".join(matrix[-1]))
    return 1


def dictlist2file(dic, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for k in dic:
            stringrow = str(k) + "\t"
            for v in dic[k]:
                stringrow += str(v) + "\t"
            stringrow = re.sub("\t$", "\n", stringrow)
            f_out.write(stringrow)
    return 1


def setuple2file(setuple, fileout, wra='w'):
    with open(fileout, wra) as f_out:
        for tup in setuple:
            stringrow = ''
            for elem in tup:
                stringrow += str(elem) + "\t"
            #print(stringrow)
            stringrow = re.sub("\t$", "\n", stringrow)
            f_out.write(stringrow)
    return 1


def writebin(data, f_out):
    out = open(f_out, "wb")
    pickle.dump(data, out)
    out.close()
    return 1


def readbin(f_in, enc="Latin-1"):
    inp = open(f_in, "rb")
    out = pickle.load(inp, encoding=enc)
    inp.close()
    return out


###############################################################################


def read_file(filename, code='utf-8'):
    with open(filename, 'r', encoding=code) as f_in:
        out = f_in.read()
        return out


def readjson(pathname):
    with open(pathname) as f_in: out = json.load(f_in)
    return out


def writejson(data, pathname):
    with open(pathname, 'w') as out: json.dump(data, out)
    return 1


def printjson(data):
    print(json.dumps(data, indent=4))
    return 1


def get_methods(obj):
    for method in dir(obj):
        if callable(getattr(obj, method)):
            print(method)
    return 1


def get_attributes(obj):
    for attribute in dir(obj):
        if not callable(getattr(obj, attribute)):
            print(attribute)
    return 1


def get_parameters(fun):
    print(fun.__doc__)
    return 1


###############################################################################


def print_matrix(matrix, lastrow=None, lastcol=None):
    for index, row in enumerate(matrix):
        if index == lastrow: break
        print(index, "-> ", end='')
        for elem in row[:lastcol]:
            print(elem, "\t", end='')
        print()
    return 1


def printdict(k2v, last=-2):
    for i, k in enumerate(k2v):
        if i == last-1: break
        print("{} \t->\t{}".format(k, k2v[k]))
    return 1


def print_dictlist(dictionary, lastrow=None, lastcol=None):
    if lastrow == None: lastrow = len(dictionary)
    ir = 0
    for index in dictionary:
        ir += 1
        if ir > lastrow: break
        print(index, "\t->\t", end='')
        ic = 0
        if lastcol == None:
            thiscol = len(dictionary[index])
        else:
            thiscol = lastcol
        for elem in dictionary[index]:
            ic += 1
            if ic > thiscol: break
            print(elem, "\t", end='')
        print()
    return 1


def print_dictset(dictionary, lastrow=None, lastcol=None):
    if lastrow == None: lastrow = len(dictionary)
    ir = 0
    for index in dictionary:
        ir += 1
        if ir > lastrow: break
        print(index, "\t->\t", end='')
        ic = 0
        if lastcol == None:
            thiscol = len(dictionary[index])
        else:
            thiscol = lastcol
        for elem in dictionary[index]:
            ic += 1
            if ic > thiscol: break
            print(elem, "\t", end='')
        print()
    return 1


###############################################################################
def print_args(args, align_size=15): # default value per i print che non appartengono ad args, tipo 'GPU in use'
    print('#'*80)
    maxlen = max([len(arg) for arg in vars(args)])
    align_size = maxlen + 3 if maxlen > align_size - 3 else align_size # garantisco ci siano sempre almeno 3 punti e lo spazio tra arg e relativo valore
    for arg in vars(args): print(f"{arg:.<{align_size}} {getattr(args, arg)}")
    return align_size


def str_or_none(value):
    if value == 'None':
        return None
    return value


def timeprogress(startime, i, step=10000, end=10000):
    i += 1
    if i % step == 0 or i == end: print(i, stringtime(startime), flush=True)
    return 1


def say_progress(n, step=10000):
    if n % step == 0: say('step', n, 'done')
    return 1


###############################################################################


def bip():
    duration = 3  # seconds
    freq = 440  # Hz
    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('say "Bernarda, sto facendo la doccia."')
    return 1


###############################################################################
import smtplib, socket, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def sendmail(dest="", sub='nerd-mail', body='nerd-mail'):
    try:
        mitt = ""
        mess = MIMEMultipart()
        mess['From'] = mitt
        mess['To'] = dest
        mess['Subject'] = sub
        mess.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('', 587)
        server.starttls()
        server.login(mitt, "")
        text = mess.as_string()
        server.sendmail(mitt, dest, text)
        server.quit()
    except (HTTPError, URLError, socket.error, ssl.SSLError, smtplib.SMTPException) as e:
        print("{}:\nmail non inviata\nerror:".format(time.strftime("%Y/%m/%d %H:%M:%S"), str(e)))


def sendslack(text='empty mess', blocks=None, attachments=None, thread_ts=None, mrkdwn=True):
    webhook_url = ""
    mess_payload = json.dumps({'text': text, 'blocks': blocks, 'attachments': attachments, 'thread_ts': thread_ts, 'mrkdwn': mrkdwn})
    os.system("curl -X POST -H 'Content-type: application/json' --data '{}' {}".format(mess_payload, webhook_url))
    print(' slack sent')
    return 1


###############################################################################


class log:
    def __init__(self, *args):
        self.timestamp = time.strftime("%y%m%d%H%M%S/")
        log_name = re.sub('.py', '.log', args[0])
        self.path = re.sub("\.py", "", args[0]) + '/'
        self.pathtime = self.path + self.timestamp
        os.system('mkdir -p ' + self.pathtime)
        for arg in args: os.system(f"cp {arg} {self.pathtime}")
        self.terminal = sys.stdout
        self.log_out = open(self.pathtime + log_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log_out.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        # sendslack('azz')
        pass


def yupdir(name=None):
    dirout = 'jupyter_' + name + time.strftime("%y%m%d%H%M%S/") if name else 'jupyter' + time.strftime("%y%m%d%H%M%S/")
    os.system('mkdir -p ' + dirout)
    print('created dirout:', dirout)
    return dirout


class bcolors:
    fucsia    = '\033[95m'
    red       = '\033[91m'
    blue      = '\033[94m'
    green     = '\033[92m'
    bold      = '\033[1m'
    underline = '\033[4m'
    end       = '\033[0m'

    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'