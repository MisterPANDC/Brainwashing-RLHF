# Data of Brainwashing-RLHF

## Hate Speech Detection Model
Please refer to [mrp](https://github.com/alatteaday/mrp_hate-speech-detection)

Download the pretrained weights:
```
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cHpBFWFWq8-o6vLFAbDVcm5Mt2SZY_l8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cHpBFWFWq8-o6vLFAbDVcm5Mt2SZY_l8" \
    -O ./Data/finetune_2nd.tar.gz && rm -rf /tmp/cookies.txt
```

To unzip the downloaded `finetune_2nd.tar.gz`, use:
```
tar -xzvf finetune_2nd.tar.gz
```

