import ast

class HitObject:
    def __init__(self,x=0,y=0,time=0,type=0,csstype="None",hitSound=0,addition=[0,0,0,0,""],
                 slidertype="",curvePoints=[],repeat=0,pixelLength=0.0,
                 edgeHitsounds=[],edgeAdditions=[],endTime=0):
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.csstype = csstype
        self.hitSound = hitSound
        self.addition = addition
        self.slidertype = slidertype
        self.curvePoints = curvePoints
        self.repeat = repeat
        self.pixelLength = pixelLength
        self.edgeHitsounds = edgeHitsounds
        self.edgeAdditions = edgeAdditions
        self.endTime = endTime

class Beatmap:
    def __init__(self):
        self.AudioFilename = ""
        self.AudioLeadIn = 0
        self.PreviewTime = 0
        self.Countdown = False
        self.SampleSet = ""
        self.StackLeniency = 0.0
        self.Mode = 0
        self.LetterboxInBreaks = False
        self.EpilepsyWarning = False
        self.WidescreenStoryboard = False

        self.Bookmarks = []
        self.DistanceSpacing = 0.0
        self.BeatDivisor = 0
        self.GridSize = 0
        self.TimelineZoom = 0

        self.Title = ""
        self.TitleUnicode = ""
        self.Artist = ""
        self.ArtistUnicode = ""
        self.Creator = ""
        self.Version = ""
        self.Source = ""
        self.Tags = []
        self.BeatmapID = 0
        self.BeatmapSetID = 0
        
        self.HPDrainRate = 0.0
        self.CircleSize = 0.0
        self.OverallDifficulty = 0.0
        self.ApproachRate = 0.0
        self.SliderMultiplier = 1.0
        self.SliderTickRate = 0.0
        
        self.Events = []
        
        self.TimingPoints = []
        
        self.Combo1 = []
        self.Combo2 = []
        self.Combo3 = []
        self.Combo4 = []
        
        self.HitObjects = []

    def calculate_slider_durations(self):
        tbb = 300
        itv = -100
        tpindex = 0
        for ho in self.HitObjects:
            if ho.csstype == "slider":
                while tpindex < len(self.TimingPoints) and self.TimingPoints[tpindex][0] <= ho.time:
                    if self.TimingPoints[tpindex][6]:
                        tbb = self.TimingPoints[tpindex][1]
                    else:
                        itv = self.TimingPoints[tpindex][1]
                    tpindex += 1
                ho.endTime = int((tbb * itv * ho.pixelLength)//(-10000 * self.SliderMultiplier))
        

def bitfield(n):
    return [1 if digit=='1' else 0 for digit in bin(n)[2:]][::-1]

def convertType(string):
    try:
        return ast.literal_eval(string)

    except ValueError:
        return string
    except SyntaxError:
        return string

def readLine(line):
    return [line.split(":")[0].strip(),line.split(":")[1].strip()]

def readBeatmap(file_location):
    bm = Beatmap()
    section = ""
    with open (file_location, 'rt', encoding="utf8") as f:
        for l in f:
            ls = l.strip()
            if len(ls) > 0:
                if ls[0] == "[":
                    section = ls
                else:
                    if section == "[General]":
                        if type(convertType(readLine(ls)[1])) == str:
                            exec("bm." + readLine(ls)[0] + " = '''" + readLine(ls)[1] + "'''")
                        else:
                            exec("bm." + readLine(ls)[0] + " = " + readLine(ls)[1])
##                    elif section == "[Editor]":
##                        if readLine(ls)[0] == "Bookmarks":
##                            bm.Bookmarks = readLine(ls)[1].split(",")
##                        elif type(convertType(readLine(ls)[1])) == str:
##                            exec("bm." + readLine(ls)[0] + " = '''" + readLine(ls)[1] + "'''")
##                        else:
##                            exec("bm." + readLine(ls)[0] + " = " + readLine(ls)[1])
##                    elif section == "[Metadata]":
##                        if readLine(ls)[0] == "Tags":
##                            bm.Tags = readLine(ls)[1].split(" ")
##                        elif type(convertType(readLine(ls)[1])) == str:
##                            exec("bm." + readLine(ls)[0] + " = '''" + readLine(ls)[1] + "'''")
##                        else:
##                            exec("bm." + readLine(ls)[0] + " = " + readLine(ls)[1])
##                    elif section == "[Difficulty]":
##                        if type(convertType(readLine(ls)[1])) == str:
##                            exec("bm." + readLine(ls)[0] + " = '''" + readLine(ls)[1] + "'''")
##                        else:
##                            exec("bm." + readLine(ls)[0] + " = " + readLine(ls)[1])
##                    elif section == "[Events]":
##                        bm.Events.append(ls)
                    elif section == "[TimingPoints]":
                        bm.TimingPoints.append([convertType(t) for t in ls.split(",")])
##                    elif section == "[Colours]":
##                        exec("bm." + readLine(ls)[0] + " = [" + readLine(ls)[1] + "]")
##                    elif section == "[HitObjects]":
##                        lss = [convertType(t) for t in ls.split(",")]
##                        typebits = bitfield(lss[3])
##                        if typebits[0] == 1:
##                            bm.HitObjects.append(HitObject(lss[0],lss[1],lss[2],lss[3],"circle",lss[4],[convertType(t) for t in lss[5].split(":")]))
##                        elif typebits[1] == 1:
##                            bm.HitObjects.append(HitObject(lss[0],lss[1],lss[2],lss[3],"slider",lss[4],None,
##                                                           lss[5].split("|")[0],[[convertType(i) for i in t.split(":")] for t in lss[5].split("|")[1:]],
##                                                           lss[6],lss[7],None,None))
##                            try:
##                                bm.HitObjects[-1].edgeHitsounds = [convertType(t) for t in lss[8].split("|")]
##                                bm.HitObjects[-1].edgeAdditions = [[convertType(i) for i in t.split(":")] for t in lss[9].split("|")]
##                                bm.HitObjects[-1].addition = [convertType(t) for t in lss[10].split(":")]
##                            except IndexError:
##                                pass
##                        elif typebits[3] == 1:
##                            bm.HitObjects.append(HitObject(lss[0],lss[1],lss[2],lss[3],"spinner",lss[4],[convertType(t) for t in lss[6].split(":")], endTime=lss[5]))
##    bm.calculate_slider_durations()
    return bm
