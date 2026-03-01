class BytePairEncoder:
    def text_to_unicode(self, text):
        return list(map(int, text.encode("utf-8")))
    
    def get_stats(self, unicode_value):
        counts = {}

        for pair in zip(unicode_value, unicode_value[1:]):
            counts[pair] = counts[pair] + 1 if pair in counts else 1

        return counts

    def merge(self, int_list, target_pair, replace):
        new_list = []
        i = 0

        while i < len(int_list):
            if i < len(int_list) - 1 and int_list[i] == target_pair[0] and int_list[i+1] == target_pair[1]:
                new_list.append(replace)
                i += 2
            else:
                new_list.append(int_list[i])
                i += 1

        return new_list

    def bpe(self, merge_list, target_size):
        num_merges = target_size - 256

        merges = {}
        for i in range(num_merges):
            stats = self.get_stats(merge_list)
            top_pair = max(stats, key=stats.get)
            new_token_id = 256 + i
            merge_list = self.merge(merge_list, top_pair, new_token_id)
            # print(f"{top_pair} -> {new_token_id}")
            merges[top_pair] = new_token_id
        
        return (merge_list, merges)

text = r"""Domestication
Wild potato species occur from the southern United States to southern Chile.[15] The potato was first domesticated in southern Peru and northwestern Bolivia[16] by pre-Columbian farmers, around Lake Titicaca.[17] Potatoes were domesticated there about 7,000–10,000 years ago from a species in the S. brevicaule complex.[16][17][18]

The earliest archaeologically verified potato tuber remains have been found at the coastal site of Ancon (central Peru), dating to 2500 BC.[19][20] The most widely cultivated variety, Solanum tuberosum tuberosum, is indigenous to the Chiloé Archipelago, and has been cultivated by the local indigenous people since before the Spanish conquest.[13][21]

Spread
Following the Spanish conquest of the Inca Empire, the Spanish introduced the potato to Europe in the second half of the 16th century as part of the Columbian exchange. The staple was subsequently conveyed by European mariners (possibly including the Russian-American Company) to territories and ports throughout the world, especially their colonies.[22] European and colonial farmers were slow to adopt farming potatoes. However, after 1750, they became an important food staple and field crop[22] and played a major role in the European 19th century population boom.[18] According to conservative estimates, the introduction of the potato was responsible for a quarter of the growth in Old World population and urbanization between 1700 and 1900.[23] However, lack of genetic diversity, due to the very limited number of varieties initially introduced, left the crop vulnerable to disease. In 1845, a plant disease known as late blight, caused by the fungus-like oomycete Phytophthora infestans, spread rapidly through the poorer communities of western Ireland as well as parts of the Scottish Highlands, resulting in the crop failures that led to the Great Irish Famine.[24][22]

The International Potato Center, based in Lima, Peru, holds 4,870 types of potato germplasm, most of which are traditional landrace cultivars.[25] In 2009, a draft sequence of the potato genome was made, containing 12 chromosomes and 860 million base pairs, making it a medium-sized plant genome.[26]

It had been thought that most potato cultivars derived from a single origin in southern Peru and extreme Northwestern Bolivia, from a species in the S. brevicaule complex.[16][17][18] DNA analysis however shows that more than 99% of all current varieties of potatoes are direct descendants of a subspecies that once grew in the lowlands of south-central Chile.[27]

Most modern potatoes grown in North America arrived through European settlement and not independently from the South American sources. At least one wild potato species, S. fendleri, occurs in North America; it is used in breeding for resistance to a nematode species that attacks cultivated potatoes. A secondary center of genetic variability of the potato is Mexico, where important wild species that have been used extensively in modern breeding are found, such as the hexaploid S. demissum, used as a source of resistance to the devastating late blight disease (Phytophthora infestans).[24] Another relative native to this region, Solanum bulbocastanum, has been used to genetically engineer the potato to resist potato blight.[28] Many such wild relatives are useful for breeding resistance to P. infestans.[29]

Little of the diversity found in Solanum ancestral and wild relatives is found outside the original South American range.[30] This makes these South American species highly valuable in breeding.[30] The importance of the potato to humanity is recognised in the United Nations International Day of Potato, to be celebrated on 30 May each year, starting in 2024.[31]

Breeding
Potatoes, both S. tuberosum and most of its wild relatives, are self-incompatible: they bear no useful fruit when self-pollinated. This trait is problematic for crop breeding, as all sexually produced plants must be hybrids. The gene responsible for self-incompatibility, as well as mutations to disable it, are now known. Self-compatibility has successfully been introduced both to diploid potatoes (including a special line of S. tuberosum) by CRISPR-Cas9.[32] Plants having a 'Sli' gene produce pollen which is compatible to its own parent and plants with similar S genes.[33] This gene was cloned by Wageningen University and Solynta in 2021, which would allow for faster and more focused breeding.[32][34]

Diploid hybrid potato breeding is a recent area of potato genetics supported by the finding that simultaneous homozygosity and fixation of donor alleles is possible.[35] Wild potato species useful for breeding blight resistance include Solanum desmissum and S. stoloniferum, among others.[36]"""

bpe = BytePairEncoder()
raw_unicode = bpe.text_to_unicode(text)
tokens, merge_list = bpe.bpe(raw_unicode, 276)