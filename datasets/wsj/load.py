# if it doesn't work, export PATH=$PATH:<path where you installed ffmpeg> 
from pydub import AudioSegment
audio = AudioSegment.from_file("data/test.wv2", "nistsphere").get_array_of_samples()

# prints the sum
loop_sum = 0
for ea_sample in audio:
	loop_sum += ea_sample
print(int(loop_sum))

# also prints the sum
print(int(sum(audio)))

# print the first 30 samples
i = 0
for ea_sample in audio:
	if i > 30:
		break
	print(ea_sample)
	i += 1
