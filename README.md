# English  ccent Detector

A tool that annlyzes p speaker's accent from a video tohhelp evaluate English language proficiency for hiringnpurposes.

##Fues

V PocessAccptpubli voURL (YuTub,Lom, odtMP4 links
 Audio trcExtcudrom video udio nmlys se video for analysis
- **AccenttClassification**: Ideitifies the speakaers 's lisE accent (e.g., Britigh, American, Ausornfnan)
- **Ccefidence  Scoingring**: iPrsoa cvidesen a score  configdenc e score for E0-100%)nglish proficiency (0-100%)
--**Explanation**: Generatessa brief explanaeian nf accent characteristics

## Demo

You can try ththle l ieve atd emo ena t: [Acor cppenht Detecator tApepe](orhttrtaplitsa://a ccoent -diseit a ecactorl.strleamlit.app/) (Note: This is a placeholder link)

##  Hwow Itorks Works

1. **Vidio doeo Download*e t*:  The toaoslt d ownlo oroads e thoei vi eo 
2. **Audio Ex Exctiontra It ectiaotsnt*e a*di: tr cIt rextraectsd the audio track from the video
3. **Transcription**: OpenAI's Whisper API transcr bet tha cperci be se e
4.  **Aceent Aallysss*: A  mccine hinre le marnl inag mod teela ain alayurzes th e audcio n
5. **Results**: The tool displays tth eete ted dccentt ecntiedn a sccrecent, econfnii

##  eetu  Inscn

### Prerequisste

- Pyttonhon  3r.8i or 
highere st n ur sst
 pe  e for transcription

tlai

 Cle t esit
  
  gi cetsreneede
 ndeecor


nsta he rii

pi sluinst
   
- FFmpeg installed on your system
-  Opeatena A.enI  Aile PIth you  kpeney  fo re Whisper transcription
   
### Insetallxampleation
   
   The ei the . l t d  pen e
1. Clone this repository:
   ```n tection
   git clone https://github.com/yourusername/accent-detector.git
  n  ce Stredmlit a p aocacly:
cent-detector
 tr am it r`` `

2. Install the required dependencies:
 he a  `i``o  e vie  http://ocahs5

   psipe install -r requirements.txt
   ```
 te ui ieo i the ted
3.  Cire ateal ae  `.ennv` file with your OpenAI API key:
   ``it f` the naisto mte
4   ie  the resclp  .eoning thv d.tectee accentx aonfidence mcorep led  .planationenv
   ```
   Te Tnheal netai edit the `.env` file to add your OpenAI API key.

# ## ideo Rrunensing th eses  tAppp to licaltai onesr aris atms
 udo racon se mpe  u r aud oesin
Run tpeehe R Sognitiotrea mses liet app  hlocea lol hy:ht tansco
 ent lassiition: rte a leor odel to e pad ith a trained ML model)

## Deployment Options

This project includes several deployment options:

1. **Docker**: Use `docker-compose up` to run the application in a contane
2. **Kubernetes**: Confguration fils in the `kubernetes/` directory
3. **AWS**: Terraform configuration in the `terraform/` directory
4. **Heroku**: Procfile and runtime.txt for Heroku deployment
5. **Google Cloud**: Deployment instructions in DEPLOYENT.md
6. **Azure**: Deployment instructions in DEPOYMENT.d

Fr detailed depoyment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md file.
```
st Limrtateona

m Thelcuite r iun apenp.tionpyse laeol  accent detection
` ``i idepend o video ent  se o
n sprs accentetetin
The application will be available at http://localhost:8501
 te ve
## Usage
 mmar e ar el for ndetin
1 . E ntero t ao  pube lic vaies dedo URL isn the input field
2 . Cloveicrocek "nAnalyze  Adceceicntn"
3. Wa itt f oro esthne  oa nualysie s teos complete
4. View the results showing the detected accent, confidence score, and explanation
 ies
## Technical Details


- **Video Processing**: Uses yt-dlp to download videos from various platforms
- **Audio Extraction**: Uses ffmpeg and pydub for audio processing
- **Speech Recognition**: Uses OpenAI's Whisper for high-quality transcription
- **Accent Classification**: Currently uses a placeholder model (to be replaced with a trained ML model)

## Deployment Options

This project includes several deployment options:

1. **Docker**: Use `docker-compose up` to run the application in a container
2. **Kubernetes**: Configuration files in the `kubernetes/` directory
3. **AWS**: Terraform configuration in the `terraform/` directory
4. **Heroku**: Procfile and runtime.txt for Heroku deployment
5. **Google Cloud**: Deployment instructions in DEPLOYMENT.md
6. **Azure**: Deployment instructions in DEPLOYMENT.md

For detailed deployment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md) file.

## Limitations

- The current implementation uses a placeholder for accent detection
- Processing time depends on video length and server load
- Only supports English accent detection

## Future Improvements

- Implement a proper machine learning model for accent detection
- Add support for more languages and accents
- Improve processing speed and efficiency
- Add batch processing for multiple videos

## License

MIT
