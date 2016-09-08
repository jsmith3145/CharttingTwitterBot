import numpy as np

remove_strings = ['COMEX']

def genetate_first_segment():
    """ Build the text for the first part of the text section    """
    sentiment = np.random.randint(0, 100)/100.0
    # be postive
    if sentiment > .5:
        lst = ['Bullish indeed, ', 'Starting to look bullish, ', 'Positive looking, ',
               'Good looking chart, ', 'Wow, ', 'Just found ', ""]
    else:
        # debbie downer
        lst = ["Look out below! ", "Yikes, ", "This could get ugly. ",
               "Not sure how this will play out... ", "Bearish! ",]

    return lst[np.random.randint(0,len(lst))]


def generate_second_segment(label):
    """ Build the text for the second part of the text section, conditionally add some additional text    """
    txt = ""
    x = np.random.randint(0, 100)/100.0
    if x > .5:
        lst = ['may be a ', "possibly a ", "looks like a "]
        txt += lst[np.random.randint(0,len(lst))]

    txt += 'head and shoulders in {}'.format(label)
    return txt

# for x in np.arange(0,10):
#     print genetate_first_segment() + generate_second_segment()

def get_hs_text(label):
    """
    Build random text, a lot more can be done here.
    >>> label = 'COMEX Silver'
    >>> get_hs_text(label)
    """
    txt = (genetate_first_segment() + generate_second_segment(label))
    for remove in remove_strings:
        txt = txt.replace(remove, "")
    return txt