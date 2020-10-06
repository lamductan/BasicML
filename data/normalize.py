import sys
def normalize(input):
	fin = open(input, "r")
	fout = open(input[:input.find('.')] + '2.data', "w")
	i = 0
	for line in fin:
		tmp = str(i)
		tmp += ','
		tmp += line
		fout.write(tmp)
		i+=1
	fout.write('\n')

def normalize2(input):
	fin = open(input, "r")
	fout = open(input[:input.find('.')] + '2.data', "w")
	i = 0
	for line in fin:
		line = line.strip('\n')
		data = line.split(',')
		tmp = str(i)
		tmp += ','
		for j in range(1,len(data)):
			tmp += data[j]
			tmp += ','
		tmp += data[0]
		tmp += '\n'
		fout.write(tmp)
		i+=1


def normalize3(input):
	fin = open(input, "r")
	fout = open(input[:input.find('.')] + '3.data', "w")
	i = 0
	for line in fin:
		line = line.strip('\n')
		data = line.split(',')
		tmp = ""
		for j in range(1,len(data)):
			tmp += data[j]
			tmp += ','
		tmp += data[0]
		tmp += '\n'
		fout.write(tmp)
		i+=1


if __name__ == '__main__':
	if len(sys.argv) > 1:
		normalize3(sys.argv[1])
