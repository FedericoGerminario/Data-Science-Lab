# This is a sample Python script.
import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def countwords(string):
    n = 0
    for el in string:
        if el + ' ' or ' ' + el:
            n = n + 1

    return n



def main():
    text = 'Non chiamatemi Bubu, implora Evani nel titolo della ' \
           'sua autobiografia, alludendo al soprannome dell&#39;' \
           'orsetto dei cartoni animati che gli venne affibbiato ' \
           'quando arriv&ograve; adolescente al Milan, per via ' \
           'della zazzera chiara, e che lo perseguit&ograve; per' \
           ' l&#39;intera e non certo banale carriera da calciatore.' \
           ' Nessuno lo chiamer&agrave; pi&ugrave; Bubu, dopo le prime ' \
           'due felici apparizioni della sua parentesi da commissario' \
           ' tecnico di scorta. In fondo &egrave; proprio lui, se si ' \
           'estremizza il concetto, la metafora vivente della Nazionale' \
           ' di Mancini: cos&igrave; precisa, nelle sua identit&agrave; ' \
           'tecnica e tattica, da potersi permettere anche contro ' \
           '<a class=\"articolo-from-repubblica\" href=\"https://www.repubblica.it/sport/calcio/nazionale/2020/11/15/news/nations_league_italia_polonia_2-0-274514406/\">un&#39;' \
           'avversaria non banale come la Polonia</a> (era in testa al girone ' \
           'e nelle 4 partite precedenti della Nations League mai era stata' \
           ' dominata, n&eacute; aveva subito pi&ugrave; di un gol,' \
           ' anzi nelle ultime due non ne aveva incassato nemmeno uno) ' \
           'di fare a meno di almeno cinque titolari senza accorgersene e ' \
           'di fare a meno pure del commissario tecnico in carica, sempre appiedato ' \
           'dal coronavirus.<br />\n<br />\nSi tratta ovviamente di un paradosso,' \
           ' perch&eacute; anche da casa Mancini ha evidentemente guidato con mano' \
           ' decisa la squadra in questi giorni di ritiro a Coverciano.' \
           ' E la squadra lo segue con soddisfazione reciproca: diventa' \
           ' difficile isolare il classico migliore in campo di questa ' \
           'partita, perch&eacute; i migliori in campo, come succede quando' \
           ' un gruppo funziona, sono stati praticamente tutti.<br />\n<br />' \
           '\nManca ora l&#39;ultimo passo per potere archiviare come anno' \
           ' buono per la Nazionale anche questo 2020 pessimo per quasi tutto' \
           ' il resto: la vittoria in Bosnia, nel vecchio stadio di Sarajevo,' \
           ' che farebbe molta pi&ugrave; paura se fosse pieno e se gli ' \
           'avversari, squassati anche da battaglie politiche extracalcistiche,' \
           ' non fossero reduci dalla doppia batosta della <a class=\"articolo-from-repubblica\"' \
           ' href=\"https://www.repubblica.it/sport/calcio/esteri/2020/11/12/news/euro_2020_fanno_festa_ungheria_scozia_e_slovacchia_fuori_la_serbia-274170507/\">' \
           'mancata qualificazione all&#39;Europeo</a> e della retrocessione ormai' \
           ' aritmetica nella Lega 2 della Nations League. Ma anche cos&igrave;, secondo' \
           ' regola aurea dello sport, le partite vanno prima giocate e poi eventualmente' \
           ' festeggiate. Se succeder&agrave;, da gioved&igrave; l&#39;Italia di Mancini sar&agrave;' \
           ' entrata finalmente in una nuova dimensione, che poi per la Nazionale, col suo status di grande del ' \
           'pallone, &egrave; in realt&agrave; la dimensione vecchia e consueta: quella di nobile del calcio mondiale. ' \
           'Nella Final Four di Nations League organizzata in casa, a Milano e a Torino nell&#39;ottobre 2021, avrebbe la possibilit&agrave; di misurarsi con la Francia campione del mondo, probabilmente col Belgio che in Russia era finito sul podio e con una tra Germania e Spagna, per le quali parla la storia.<br />\n<br />\nPrima ci saranno il sorteggio del 7 dicembre (di nuovo da testa di serie e alcuni reduci dalla notte di tre anni fa a San Siro con la Svezia sanno bene che cosa possa significare non esserlo), l&#39;inizio delle qualificazioni e l&#39;Europeo di giugno. Dove l&#39;Italia, con un anno in pi&ugrave; di crescita e una rosa che si &egrave; arricchita via via di alternative tutt&#39;altro che di secondo piano come Locatelli, Bastoni e Berardi, non pu&ograve; pi&ugrave; nascondere il ruolo di favorita aggiunta. Per diventarlo a pieno titolo, manca forse ancora il gol del centravanti. Ma per lo pi&ugrave; non manca molto, <a class=\"articolo-from-repubblica\" href=\"https://www.repubblica.it/sport/calcio/nazionale/2020/11/15/news/nazionale_evani_uniti_nelle_difficolta_questi_ragazzi_sono_straordinari_-274513956/\">ha sottolineato Evani</a>, che ha parlato anche di squadra quasi perfetta. Qualche ora prima il presidente della Figc Gravina aveva raccontato di essere <a class=\"articolo-from-repubblica\" href=\"https://www.repubblica.it/sport/calcio/nazionale/2020/11/15/news/nazionale_gravina_visita_ospedale_reggio_contratto_mancini-274476923/\">in attesa della risposta di Mancini</a> alla sua proposta di rinnovo del contratto, che scade nel 2022. Da gioved&igrave;, se l&#39;Italia sar&agrave; tornata a tutti gli effetti tra le grandi, sar&agrave; il nuovo tema sul tavolo azzurro.'
    clean_text = cleanhtml(text)
    print(clean_text)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
