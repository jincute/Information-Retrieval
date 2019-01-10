# To convert the TREC topics in to Indri Query parameter xml file
import glob
import re
from os.path import basename, dirname
import sys
import relation_extractor_mac as relex


def gen_query_text(query_text):
    if QUERY_EXPANSION:
        file_json = "/Users/jeanneluo/UdeM/IFT6255_RI/devoir1/Code/ap_cfd_dist5_min10_top10_stp.json"
        relex.reload_cfd_json(file_json)
        expans_list = relex.get_exapn_for_query(query_text.lower())
        if len(expans_list) > 0:
            expans_weight_query = "#weight( " + ''.join(['{:.3f} {} '.format(s, w) for w, s in expans_list.items()])+')'
            query_text_expansion = "#weight({:} #combine({}) {:} {})".format(ORIGINAL_WEIGHT, query_text,
                                                                             1 - ORIGINAL_WEIGHT, expans_weight_query)
        else:
            query_text_expansion = "#combine( {} )".format(query_text)
        r = query_text_expansion
    else:
        r = "#combine( {} )".format(query_text)
    return r


def convert_topic():
    if len(sys.argv) > 1:
        # Input File path
        path = sys.argv[1]

    # Output file will be lcated in the same path as input file
    # but with the output_prefix and extension .xml
    output_prefix = "query_param_"
    list_of_file = glob.glob(path)
    # Dictionary for abbreviation with the dot e.g.  U.S.
    dict_abbrev = {"U.S.": "USA"}

    for file in list_of_file:
        lists_topics = []
        with open(file) as fp:
            num_val = 0
            for line in fp:
                # Get <num>
                result = re.match(r"<num>\s*Number:\s*(\d+)\s*\n", line)
                if bool(result):
                    num_val = int( result.group(1), 10)

                # Get <title>
                title_str = ""
                result = re.match(r"<title>\s*Topic:\s*(.*)\n", line)
                if bool(result):
                    title_str = result.group(1)
                    # Replace U.S. by  United States for stupid indri
                    # for word in dict_abbrev:
                    #     title_str = title_str.replace(word, dict_abbrev[word])
                    # Enlever les points(.)
                    # Remplacer les gillemets(") et traits(-) et tous les autres avec une espace.
                    # Donc les mots comme  U.S.-U.S.S.R. deviendront US USSR
                    # To avoid cause error in Indri 5.12
                    title_str = title_str.replace(".", "")
                    # re.sub(r"[^A-Za-z]+", ' ', title_str) #Just keep english letters.
                    # Replace all the non-alphas with space, but keep digits
                    title_str = ''.join([(x if x.isalpha() or x.isdigit() else ' ') for x in title_str])
                    lists_topics += [{"num": num_val, "title": title_str}]

        print("{} ({} request converted)".format(file, len(lists_topics)))

        # Write the Indri query format file
        with open(dirname(path) + "/" + output_prefix + basename(file)[:-4] + ".xml", "w") as fw:
            param = ("<parameters>\n")
            for topic in lists_topics:
                query = (
                        "\t<query>\n"
                        "\t\t<type>indri</type>\n"
                        "\t\t<number>{0}</number>\n"
                        "\t\t<text>\n"
                        "\t\t\t{1}\n"
                        "\t\t</text>\n"
                        "\t</query>\n"
                         )
                param += query.format(topic["num"], gen_query_text(topic["title"]) )
            param += ("\t<memory>1G</memory>\n"
                         "\t<index>./test_output</index>\n"
                         "\t<trecFormat>true</trecFormat>\n"
                         "\t<count>1000</count>\n"
                         "</parameters>")
            fw.writelines(param)


QUERY_EXPANSION = True
ORIGINAL_WEIGHT = 0.75

if __name__ == "__main__":
     convert_topic()
