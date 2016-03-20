#! /bin/bash
cd data
if [ ! -f attributes-half.json ]; then
    wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/attributes-half.json.zip
     unzip attributes-half.json.zip
     rm attributes-half.json.zip
fi
if [ ! -f image_data.json ]; then
     wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/image_data.json.zip
     unzip image_data.json.zip
     rm image_data.json.zip
fi
if [ ! -f objects.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/objects.json.zip
	unzip objects.json.zip
	rm objects.json.zip
fi
if [ ! -f question_answers.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/question_answers.json.zip
	unzip question_answers.json.zip
	rm question_answers.json.zip
fi
if [ ! -f region_descriptions.json  ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/region_descriptions.json.zip
	unzip region_descriptions.json.zip
	rm region_descriptions.json.zip
fi
if [ ! -f relationships-half.json ]; then
	wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/relationships-half.json.zip
	unzip relationships-half.json.zip
	rm relationships-half.json.zip
fi
if [ ! -d images ]; then
     wget https://googledrive.com/host/0B046sNk0DhCDN3R4VzJBa0ZSSms/images.zip
     unzip images.zip -d images
     rm images.zip
fi
rm -rf __MACOSX