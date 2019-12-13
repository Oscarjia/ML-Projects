wc -l token_1.txt
sed '/^<doc / d' token_1.txt > token_1_noid.txt
wc -l token_1_noid.txt
wc -l token_2.txt
sed '/^<doc / d' token_2.txt > token_2_noid.txt
wc -l token_2_noid.txt
wc -l token_3.txt
sed '/^<doc / d' token_3.txt > token_3_noid.txt
wc -l token_3_noid.txt