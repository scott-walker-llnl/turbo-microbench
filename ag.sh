exp="exp1"
for ((n = 0; n < 200; n += 2))
{
	path=data/$exp/run$n
	grep -o "[0-9]*\.[0-9]*\sseconds" $path/info
}
