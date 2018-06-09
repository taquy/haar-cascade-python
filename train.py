import cv2
import numpy as np
import shutil
import os
from os import listdir
from os.path import isfile, join, isdir, exists
from distutils.dir_util import copy_tree
import sys

def main() :
	global POSITIVE_WIDTH, POSITIVE_HEIGHT, NEGATIVE_WIDTH, NEGATIVE_HEIGHT, SAMPLE_WIDTH, SAMPLE_HEIGHT, MAX_X_ANGLE
	global MAX_Y_ANGLE, MAX_Z_ANGLE, FEATURE_LBP, MIN_HIT_RATE, STAGES, SAMPLE_SCALE

	SAMPLE_WIDTH = str(SAMPLE_WIDTH)
	SAMPLE_HEIGHT = str(SAMPLE_HEIGHT)
	STAGES = str(STAGES)
	MAX_X_ANGLE = str(MAX_X_ANGLE)
	MAX_Y_ANGLE = str(MAX_Y_ANGLE)
	MAX_Z_ANGLE = str(MAX_Z_ANGLE)
	MIN_HIT_RATE = str(MIN_HIT_RATE)

	mp_p = 'pos'
	mp_n = 'neg'
	pfs = (POSITIVE_WIDTH, POSITIVE_HEIGHT)
	nfs = (NEGATIVE_WIDTH, NEGATIVE_HEIGHT)
	sps = (SAMPLE_WIDTH, SAMPLE_HEIGHT)

	pps = '/tmp/p' + mp_p
	nns = '/tmp/n' + mp_n
	fvec = '/tmp/if.vec'
	fdat = '/tmp/if.dat'
	ftxt = '/tmp/bg.txt'
	rslt = '/tmp/result'

	print ('Creating environment...')

	pf = [f for f in listdir(mp_p) if isfile(join(mp_p, f))]
	nf = [f for f in listdir(mp_n) if isfile(join(mp_n, f))]

	if exists(rslt): shutil.rmtree(rslt)
	if exists(pps): shutil.rmtree(pps)
	if exists(nns): shutil.rmtree(nns)

	if not exists(rslt): os.makedirs(rslt)
	if not exists(pps): os.makedirs(pps)
	if not exists(nns): os.makedirs(nns)
	print ('Creating environment done.')
	print ('Preprocessing images...')

	for f in pf:
		imgpath = mp_p + '/' + f
		fn, fe = os.path.splitext(imgpath)

		rsl = cv2.imread(imgpath)
		# rsl = cv2.cvtColor(rsl, cv2.COLOR_BGR2GRAY)

		if rsl is None: continue
		rsl = cv2.resize(rsl, pfs)
		cv2.imwrite(pps + '/' + f, rsl)

	for f in nf:
		imgpath = mp_n + '/' + f
		rsl = cv2.imread(imgpath)
		# rsl = cv2.cvtColor(rsl, cv2.COLOR_BGR2GRAY)

		if rsl is None: continue
		rsl = cv2.resize(rsl, nfs)
		cv2.imwrite(nns + '/' + f, rsl)

	print ('Preprocessing images done')
	print ('Normalize images...')

	cmd = 'cd ' + pps + '; j=1;for i in *; do mv "$i" "$j"; j=$((j+1));done;'
	os.system(cmd)

	cmd = 'cd ' + nns + '; j=1;for i in *; do mv "$i" "$j"; j=$((j+1));done;'
	os.system(cmd)

	print ('Normalize images done')
	print ('Creating info annotation...')
	print ('Creating background annotation...')

	ppf = [f for f in listdir(pps) if isfile(join(pps, f))]
	nnf = [f for f in listdir(nns) if isfile(join(nns, f))]

	s = ['','']
	for f in ppf: s[0] += pps + '/' + f + ' 1 0 0 ' + str(pfs[0]) + ' ' + str(pfs[1]) + '\n'
	for f in nnf: s[1] += nns + '/' + f + '\n'

	def save(fn, d) :
		f = open(fn,'w') 
		f.write(d) 
		f.close()

	os.remove(fvec) if os.path.exists(fvec) else None
	os.remove(fdat) if os.path.exists(fdat) else None
	os.remove(ftxt) if os.path.exists(ftxt) else None

	save(fdat, s[0])
	print ('Creating info done.')
	save(ftxt, s[1])
	print ('Creating background done.')

	# create sample positives
	cmd = 'cd /tmp; opencv_createsamples -info if.dat -num ' + str(len(pf)) * SAMPLE_SCALE + ' -w ' + str(sps[0]) + ' -h ' + str(sps[1]) + ' -vec if.vec '
	if not MAX_X_ANGLE is '' : cmd += ' -maxxangle= ' + MAX_X_ANGLE
	if not MAX_Y_ANGLE is '' : cmd += ' -maxyangle= ' + MAX_Y_ANGLE
	if not MAX_Z_ANGLE is '' : cmd += ' -maxzangle= ' + MAX_Z_ANGLE
	os.system(cmd)

	# train cascade
	cmd = 'cd /tmp; opencv_traincascade -data result -vec if.vec -bg bg.txt -numPos ' + str(len(pf) * 0.9)
	cmd += ' -numNeg ' + str(len(nf)) + ' -numStages ' + STAGES + ' -w ' + str(sps[0]) + ' -h ' + str(sps[1]) + ' '
	if FEATURE_LBP is 1 : cmd += ' -featureType LBP'
	if not MIN_HIT_RATE is '' : cmd += ' -minHitRate ' + MIN_HIT_RATE

	os.system(cmd)

	copy_tree(rslt, os.getcwd() + '/result')

	shutil.rmtree(pps)
	shutil.rmtree(nns)
	shutil.rmtree(rslt)
	os.remove(fvec)
	os.remove(fdat)
	os.remove(ftxt)


cff = 'config.txt'
cfl = []
cfld = {}

POSITIVE_WIDTH = '' 		# required
POSITIVE_HEIGHT = '' 		# required
NEGATIVE_WIDTH = '' 		# required
NEGATIVE_HEIGHT = '' 		# required
SAMPLE_WIDTH = '' 			# required
SAMPLE_HEIGHT = '' 			# required
STAGES = ''					# requried
SAMPLE_SCALE = ''					# requried
MAX_X_ANGLE = ''
MAX_Y_ANGLE = ''
MAX_Z_ANGLE = ''
FEATURE_LBP = ''
MIN_HIT_RATE = ''
MIN_HIT_RATE = ''



def isNumber(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def isFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def init() :

	global POSITIVE_WIDTH, POSITIVE_HEIGHT, NEGATIVE_WIDTH, NEGATIVE_HEIGHT, SAMPLE_WIDTH, SAMPLE_HEIGHT
	global MAX_X_ANGLE, MAX_Y_ANGLE, MAX_Z_ANGLE, FEATURE_LBP, MIN_HIT_RATE, STAGES, SAMPLE_SCALE

	if os.path.isfile(cff) :
		with open(cff) as f:
			cfl = f.readlines()
			for l in cfl:
				l = l.replace('\n', '')
				l = l.split('=')
				cfld[l[0]] = l[1]
			# validate
			if 'POSITIVE_WIDTH' in cfld : POSITIVE_WIDTH = cfld['POSITIVE_WIDTH'].strip()
			if 'POSITIVE_HEIGHT' in cfld : POSITIVE_HEIGHT = cfld['POSITIVE_HEIGHT'].strip()
			if 'NEGATIVE_WIDTH' in cfld : NEGATIVE_WIDTH = cfld['NEGATIVE_WIDTH'].strip()
			if 'NEGATIVE_HEIGHT' in cfld : NEGATIVE_HEIGHT = cfld['NEGATIVE_HEIGHT'].strip()
			if 'SAMPLE_WIDTH' in cfld : SAMPLE_WIDTH = cfld['SAMPLE_WIDTH'].strip()
			if 'SAMPLE_HEIGHT' in cfld : SAMPLE_HEIGHT = cfld['SAMPLE_HEIGHT'].strip()
			if 'FEATURE_LBP' in cfld : FEATURE_LBP = cfld['FEATURE_LBP'].strip()
			if 'MAX_X_ANGLE' in cfld : MAX_X_ANGLE = cfld['MAX_X_ANGLE'].strip()
			if 'MAX_Y_ANGLE' in cfld : MAX_Y_ANGLE = cfld['MAX_Y_ANGLE'].strip()
			if 'MAX_Z_ANGLE' in cfld : MAX_Z_ANGLE = cfld['MAX_Z_ANGLE'].strip()
			if 'MIN_HIT_RATE' in cfld : MIN_HIT_RATE = cfld['MIN_HIT_RATE'].strip()
			if 'STAGES' in cfld : STAGES = cfld['STAGES'].strip()
			if 'SAMPLE_SCALE' in cfld : SAMPLE_SCALE = cfld['SAMPLE_SCALE'].strip()

			hasError = False
			errMsg = '' 

			if POSITIVE_WIDTH is '':
				hasError = True
				errMsg += 'Error: POSITIVE_WIDTH is required \n'
			else:
				POSITIVE_WIDTH = int(POSITIVE_WIDTH)

			if POSITIVE_HEIGHT is '':
				hasError = True
				errMsg += 'Error: POSITIVE_HEIGHT is required \n'
			else:
				POSITIVE_HEIGHT = int(POSITIVE_HEIGHT)

			if NEGATIVE_WIDTH is '':
				hasError = True
				errMsg += 'Error: NEGATIVE_WIDTH is required \n'
			else:
				NEGATIVE_WIDTH = int(NEGATIVE_WIDTH)

			if NEGATIVE_HEIGHT is '':
				hasError = True
				errMsg += 'Error: NEGATIVE_HEIGHT is required \n'
			else:
				NEGATIVE_HEIGHT = int(NEGATIVE_HEIGHT)

			if SAMPLE_WIDTH is '':
				hasError = True
				errMsg += 'Error: SAMPLE_WIDTH is required \n'
			else:
				SAMPLE_WIDTH = int(SAMPLE_WIDTH)

			if SAMPLE_HEIGHT is '':
				hasError = True
				errMsg += 'Error: SAMPLE_HEIGHT is required \n'
			else:
				SAMPLE_HEIGHT = int(SAMPLE_HEIGHT)

			if STAGES is '':
				hasError = True
				errMsg += 'Error: STAGES is required \n'
			else:
				STAGES = int(STAGES)

			if SAMPLE_SCALE is '':
				hasError = True
				errMsg += 'Error: SAMPLE_SCALE is required \n'
			else:
				SAMPLE_SCALE = int(SAMPLE_SCALE)

			if hasError:
				print (errMsg)
				return

			if not isNumber(POSITIVE_WIDTH) :
				hasError = True
				errMsg += 'Error: POSITIVE_WIDTH must be number\n'

			if not isNumber(POSITIVE_HEIGHT) :
				hasError = True
				errMsg += 'Error: POSITIVE_HEIGHT must be number\n'

			if not isNumber(NEGATIVE_WIDTH) :
				hasError = True
				errMsg += 'Error: NEGATIVE_WIDTH must be number\n'

			if not isNumber(NEGATIVE_HEIGHT) :
				hasError = True
				errMsg += 'Error: NEGATIVE_HEIGHT must be number\n'

			if not isNumber(SAMPLE_WIDTH) :
				hasError = True
				errMsg += 'Error: SAMPLE_WIDTH must be number\n'

			if not isNumber(SAMPLE_HEIGHT) :
				hasError = True
				errMsg += 'Error: SAMPLE_HEIGHT must be number\n'

			if not isNumber(STAGES) :
				hasError = True
				errMsg += 'Error: STAGES must be number\n'

			if not isNumber(SAMPLE_SCALE) :
				hasError = True
				errMsg += 'Error: SAMPLE_SCALE must be number\n'

			if hasError:
				print (errMsg)
				return

			if not MAX_X_ANGLE is '':
				if not isFloat(MAX_X_ANGLE) :
					hasError = True
					errMsg += 'Error: MAX_X_ANGLE must be number\n'
				else:
					MAX_X_ANGLE = int(MAX_X_ANGLE)
			if not MAX_Y_ANGLE is '':
				if not isFloat(MAX_Y_ANGLE) :
					hasError = True
					errMsg += 'Error: MAX_Y_ANGLE must be number\n'
				else:
					MAX_Y_ANGLE = int(MAX_Y_ANGLE)
			if not MAX_Z_ANGLE is '':
				if not isFloat(MAX_Z_ANGLE) :
					hasError = True
					errMsg += 'Error: MAX_Z_ANGLE must be number\n'
				else:
					MAX_Z_ANGLE = int(MAX_Z_ANGLE)
			if not FEATURE_LBP is '':
				if not isNumber(FEATURE_LBP) :
					hasError = True
					errMsg += 'Error: FEATURE_LBP must be number\n'
				else:
					FEATURE_LBP = int(FEATURE_LBP)
			if not MIN_HIT_RATE is '':
				if not isFloat(MIN_HIT_RATE) :
					hasError = True
					errMsg += 'Error: MIN_HIT_RATE must be number\n'
				else:
					MIN_HIT_RATE = int(MIN_HIT_RATE)

			if hasError:
				print (errMsg)
				return

			if int(POSITIVE_WIDTH) >= int(NEGATIVE_WIDTH):
				hasError = True
				errMsg += 'Error: POSITIVE_WIDTH must smaller than NEGATIVE_WIDTH\n'

			if POSITIVE_HEIGHT >= NEGATIVE_HEIGHT:
				hasError = True
				errMsg += 'Error: POSITIVE_HEIGHT must smaller than NEGATIVE_HEIGHT\n'

			if hasError:
				print (errMsg)
				return

			if POSITIVE_WIDTH <= 0:
				hasError = True
				errMsg += 'Error: POSITIVE_WIDTH must be > than 0\n'

			if POSITIVE_HEIGHT <= 0:
				hasError = True
				errMsg += 'Error: POSITIVE_HEIGHT must be > than 0\n'

			if NEGATIVE_WIDTH <= 0:
				hasError = True
				errMsg += 'Error: NEGATIVE_WIDTH must be > than 0\n'

			if NEGATIVE_HEIGHT <= 0:
				hasError = True
				errMsg += 'Error: NEGATIVE_HEIGHT must be > than 0\n'

			if SAMPLE_WIDTH <= 0:
				hasError = True
				errMsg += 'Error: SAMPLE_WIDTH must be > than 0\n'

			if SAMPLE_HEIGHT <= 0:
				hasError = True
				errMsg += 'Error: SAMPLE_HEIGHT must be > than 0\n'

			if STAGES <= 0:
				hasError = True
				errMsg += 'Error: STAGES must be > than 0\n'

			if SAMPLE_SCALE <= 0:
				hasError = True
				errMsg += 'Error: SAMPLE_SCALE must be > than 0\n'

			if not FEATURE_LBP is '':
				if not (FEATURE_LBP is 0 or FEATURE_LBP is 1):
					hasError = True
					errMsg += 'Error: FEATURE_LBP must be either 0 or 1\n'

			if not MIN_HIT_RATE is '':
				if MIN_HIT_RATE <= 0:
					hasError = True
					errMsg += 'Error: MIN_HIT_RATE must be > than 0\n'

			if hasError:
				print (errMsg)
				return

			main()
	else :
		print ('Config file is not detected.')
		file = open(cff,'w')
		print ('Config.txt created.')

		file.write('POSITIVE_WIDTH=100\n')
		file.write('POSITIVE_HEIGHT=100\n')
		file.write('NEGATIVE_WIDTH=640\n')
		file.write('NEGATIVE_HEIGHT=480\n')
		file.write('SAMPLE_WIDTH=40\n')
		file.write('SAMPLE_HEIGHT=40\n')
		file.write('STAGES=10\n')
		file.write('FEATURE_LBP=1\n')
		file.write('SAMPLE_SCALE=5\n')
		file.write('MAX_X_ANGLE=\n')
		file.write('MAX_Y_ANGLE=\n')
		file.write('MAX_Z_ANGLE=\n')
		file.write('MIN_HIT_RATE=\n')
		file.close()

		print ('What would you like to do?')
		print ('1. Edit the config file')
		print ('2. Run the train script')
		print ('0. Exit')
		ipt = input('>>')
		if ipt is '2':
			os.system('python3 train.py')
		elif ipt is '1':
			os.system('gedit config.txt')
			print ('Run the script now? (y/n)')
			ipt = input('>>')
			if ipt is 'y' or ipt is 'Y':
				os.system('python3 train.py')


	

init()

# # opencv_createsamples -vec if.vec -w 34 -h 34

# # opencv_traincascade -data result -vec if.vec -bg bg.txt -numPost 1 -numNeg 1 -numStages 2 -w 34 -h 34 -featureType LBP


