exp="exp1"
for ((n = 0; n < 199; n++))
{
	path=data/$exp/run$n
	grep -o "[0-9.]*\sseconds" $path/info
}
