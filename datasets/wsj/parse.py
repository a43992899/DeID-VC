import pathlib, os

# if true, there can be multiple audio files from each speaker with
# the same transcription. I believe the audio files are identical.
REDUNDANCY = False
# format of transcription file. options: "dot", "ptx", "lsn". does not work with PTX.
# dot (sometimes) includes lip smacks and tongue clicks, lsn is all caps words.
TRANSC_FORMAT = "dot"

data = [] # (file, speaker id, transcription)
lookup = {}
transc = []

# traverse file heirarchy
def transc_dir(dir):
	global transc, lookup
	for filename in sorted(os.listdir(dir)):
		full_fn = dir + '/' + filename
		if os.path.isdir(full_fn):
			transc_dir(full_fn)
		else:
			if filename.split('.')[-1] == TRANSC_FORMAT:
				transc.append(full_fn)
			elif filename.split('.')[-1] == "wv1":# in ("wv1", "wv2"): # wv2 uses a different microphone
				stem = filename.split('.')[0]
				if stem in lookup:
					lookup[stem].append(full_fn)
				else:
					lookup[stem] = [full_fn]

import sys
if __name__ == "__main__":
	if len(sys.argv) > 1:
		dir = sys.argv[1]
	else:
		# defaults to current working directory
		dir = "."
	transc_dir(dir)
	for ea_transc_file in transc:
		with open(ea_transc_file, "r") as f:
			for ea_line in f:
				# get last word in transcription (the file stem)
				line = ea_line.split(' ')
				filestem = line.pop().strip()
				assert(filestem[0] == '(' and filestem[-1] == ')')
				filestem = filestem[1:-1]

				# add entry to `data`
				if filestem not in lookup:
				#d	print(filestem)
					continue
				elif REDUNDANCY:
					for ea_file in lookup[filestem]:
						data.append((ea_file[0], filestem[:3], ' '.join(line)))
				else:
					data.append((lookup[filestem][0], filestem[:3], ' '.join(line)))
				del lookup[filestem]

	# generate csv file
	outstring = ""
	for ea_element in data:
		outstring += ea_element[0] + "|" + ea_element[1] + "|" + ea_element[2] + "\n"
	outstring = outstring[:-1]

	# write to csv file
	with open("out.csv", "w") as f:
		f.write(outstring)
