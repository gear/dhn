if [ ! -f "./data/com-youtube.ungraph.txt.gz" ] && [ ! -f "./data/com-youtube.ungraph.txt" ]; then
    wget -P data https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
    gunzip ./data/com-youtube.ungraph.txt.gz
fi
if [ ! -f "./data/web-Google.txt.gz" ] && [ ! -f "./data/web-Google.txt" ]; then
    wget -P data https://snap.stanford.edu/data/web-Google.txt.gz
    gunzip ./data/web-Google.txt.gz 
fi
