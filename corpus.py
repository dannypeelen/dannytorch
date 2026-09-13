"""
Larger training corpus for model2 (d_model=128, n_heads=4, n_blocks=3 -> ~610k params).
The original inline CORPUS2 was ~2.6k characters, about 230x smaller than the
parameter count -- way too little data for the model to learn anything beyond
per-character frequency (that's exactly why training got stuck at the corpus's
unigram entropy, ~2.97 nats, instead of dropping further).

This corpus extends the same five public-domain works already referenced in the
original snippet (A Tale of Two Cities, Hamlet, Pride and Prejudice, Moby-Dick,
Genesis) out to their full opening chapter/scene, giving ~34.5k characters --
about 13x more text, still small enough to train in reasonable time with
dannytorch's pure-Python/numpy autograd, but enough to move past pure frequency
matching.

Sources (public domain, via Project Gutenberg):
  - Charles Dickens, "A Tale of Two Cities", Book 1, Chapter 1 ("The Period")
  - William Shakespeare, "Hamlet", Act 1, Scene 1
  - Jane Austen, "Pride and Prejudice", Chapter 1
  - Herman Melville, "Moby-Dick", Chapter 1 ("Loomings")
  - Genesis 1 (King James Version)

Non-ASCII punctuation (curly quotes, em dashes) has been normalized to plain
ASCII to keep the character vocabulary consistent with the rest of the demo.
"""

CORPUS2 = """It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it
was the epoch of incredulity, it was the season of Light, it was the
season of Darkness, it was the spring of hope, it was the winter of
despair, we had everything before us, we had nothing before us, we were
all going direct to Heaven, we were all going direct the other way--in
short, the period was so far like the present period, that some of its
noisiest authorities insisted on its being received, for good or for
evil, in the superlative degree of comparison only.

There were a king with a large jaw and a queen with a plain face, on the
throne of England; there were a king with a large jaw and a queen with
a fair face, on the throne of France. In both countries it was clearer
than crystal to the lords of the State preserves of loaves and fishes,
that things in general were settled for ever.

It was the year of Our Lord one thousand seven hundred and seventy-five.
Spiritual revelations were conceded to England at that favoured period,
as at this. Mrs. Southcott had recently attained her five-and-twentieth
blessed birthday, of whom a prophetic private in the Life Guards had
heralded the sublime appearance by announcing that arrangements were
made for the swallowing up of London and Westminster. Even the Cock-lane
ghost had been laid only a round dozen of years, after rapping out its
messages, as the spirits of this very year last past (supernaturally
deficient in originality) rapped out theirs. Mere messages in the
earthly order of events had lately come to the English Crown and People,
from a congress of British subjects in America: which, strange
to relate, have proved more important to the human race than any
communications yet received through any of the chickens of the Cock-lane
brood.

France, less favoured on the whole as to matters spiritual than her
sister of the shield and trident, rolled with exceeding smoothness down
hill, making paper money and spending it. Under the guidance of her
Christian pastors, she entertained herself, besides, with such humane
achievements as sentencing a youth to have his hands cut off, his tongue
torn out with pincers, and his body burned alive, because he had not
kneeled down in the rain to do honour to a dirty procession of monks
which passed within his view, at a distance of some fifty or sixty
yards. It is likely enough that, rooted in the woods of France and
Norway, there were growing trees, when that sufferer was put to death,
already marked by the Woodman, Fate, to come down and be sawn into
boards, to make a certain movable framework with a sack and a knife in
it, terrible in history. It is likely enough that in the rough outhouses
of some tillers of the heavy lands adjacent to Paris, there were
sheltered from the weather that very day, rude carts, bespattered with
rustic mire, snuffed about by pigs, and roosted in by poultry, which
the Farmer, Death, had already set apart to be his tumbrils of
the Revolution. But that Woodman and that Farmer, though they work
unceasingly, work silently, and no one heard them as they went about
with muffled tread: the rather, forasmuch as to entertain any suspicion
that they were awake, was to be atheistical and traitorous.

In England, there was scarcely an amount of order and protection to
justify much national boasting. Daring burglaries by armed men, and
highway robberies, took place in the capital itself every night;
families were publicly cautioned not to go out of town without removing
their furniture to upholsterers' warehouses for security; the highwayman
in the dark was a City tradesman in the light, and, being recognised and
challenged by his fellow-tradesman whom he stopped in his character of
"the Captain," gallantly shot him through the head and rode away; the
mail was waylaid by seven robbers, and the guard shot three dead, and
then got shot dead himself by the other four, "in consequence of the
failure of his ammunition:" after which the mail was robbed in peace;
that magnificent potentate, the Lord Mayor of London, was made to stand
and deliver on Turnham Green, by one highwayman, who despoiled the
illustrious creature in sight of all his retinue; prisoners in London
gaols fought battles with their turnkeys, and the majesty of the law
fired blunderbusses in among them, loaded with rounds of shot and ball;
thieves snipped off diamond crosses from the necks of noble lords at
Court drawing-rooms; musketeers went into St. Giles's, to search
for contraband goods, and the mob fired on the musketeers, and the
musketeers fired on the mob, and nobody thought any of these occurrences
much out of the common way. In the midst of them, the hangman, ever busy
and ever worse than useless, was in constant requisition; now, stringing
up long rows of miscellaneous criminals; now, hanging a housebreaker on
Saturday who had been taken on Tuesday; now, burning people in the
hand at Newgate by the dozen, and now burning pamphlets at the door of
Westminster Hall; to-day, taking the life of an atrocious murderer,
and to-morrow of a wretched pilferer who had robbed a farmer's boy of
sixpence.

All these things, and a thousand like them, came to pass in and close
upon the dear old year one thousand seven hundred and seventy-five.
Environed by them, while the Woodman and the Farmer worked unheeded,
those two of the large jaws, and those other two of the plain and the
fair faces, trod with stir enough, and carried their divine rights
with a high hand. Thus did the year one thousand seven hundred
and seventy-five conduct their Greatnesses, and myriads of small
creatures--the creatures of this chronicle among the rest--along the
roads that lay before them.

SCENE I. Elsinore. A platform before the Castle.


Enter Francisco and Barnardo, two sentinels.

BARNARDO.
Who's there?

FRANCISCO.
Nay, answer me. Stand and unfold yourself.

BARNARDO.
Long live the King!

FRANCISCO.
Barnardo?

BARNARDO.
He.

FRANCISCO.
You come most carefully upon your hour.

BARNARDO.
'Tis now struck twelve. Get thee to bed, Francisco.

FRANCISCO.
For this relief much thanks. 'Tis bitter cold,
And I am sick at heart.

BARNARDO.
Have you had quiet guard?

FRANCISCO.
Not a mouse stirring.

BARNARDO.
Well, good night.
If you do meet Horatio and Marcellus,
The rivals of my watch, bid them make haste.

Enter Horatio and Marcellus.

FRANCISCO.
I think I hear them. Stand, ho! Who is there?

HORATIO.
Friends to this ground.

MARCELLUS.
And liegemen to the Dane.

FRANCISCO.
Give you good night.

MARCELLUS.
O, farewell, honest soldier, who hath reliev'd you?

FRANCISCO.
Barnardo has my place. Give you good-night.

Exit.

MARCELLUS.
Holla, Barnardo!

BARNARDO.
Say, what, is Horatio there?

HORATIO.
A piece of him.

BARNARDO.
Welcome, Horatio. Welcome, good Marcellus.

MARCELLUS.
What, has this thing appear'd again tonight?

BARNARDO.
I have seen nothing.

MARCELLUS.
Horatio says 'tis but our fantasy,
And will not let belief take hold of him
Touching this dreaded sight, twice seen of us.
Therefore I have entreated him along
With us to watch the minutes of this night,
That if again this apparition come
He may approve our eyes and speak to it.

HORATIO.
Tush, tush, 'twill not appear.

BARNARDO.
Sit down awhile,
And let us once again assail your ears,
That are so fortified against our story,
What we two nights have seen.

HORATIO.
Well, sit we down,
And let us hear Barnardo speak of this.

BARNARDO.
Last night of all,
When yond same star that's westward from the pole,
Had made his course t'illume that part of heaven
Where now it burns, Marcellus and myself,
The bell then beating one--

MARCELLUS.
Peace, break thee off. Look where it comes again.

Enter Ghost.

BARNARDO.
In the same figure, like the King that's dead.

MARCELLUS.
Thou art a scholar; speak to it, Horatio.

BARNARDO.
Looks it not like the King? Mark it, Horatio.

HORATIO.
Most like. It harrows me with fear and wonder.

BARNARDO
It would be spoke to.

MARCELLUS.
Question it, Horatio.

HORATIO.
What art thou that usurp'st this time of night,
Together with that fair and warlike form
In which the majesty of buried Denmark
Did sometimes march? By heaven I charge thee speak.

MARCELLUS.
It is offended.

BARNARDO.
See, it stalks away.

HORATIO.
Stay! speak, speak! I charge thee speak!

Exit Ghost.

MARCELLUS.
'Tis gone, and will not answer.

BARNARDO.
How now, Horatio! You tremble and look pale.
Is not this something more than fantasy?
What think you on't?

HORATIO.
Before my God, I might not this believe
Without the sensible and true avouch
Of mine own eyes.

MARCELLUS.
Is it not like the King?

HORATIO.
As thou art to thyself:
Such was the very armour he had on
When he th'ambitious Norway combated;
So frown'd he once, when in an angry parle
He smote the sledded Polacks on the ice.
'Tis strange.

MARCELLUS.
Thus twice before, and jump at this dead hour,
With martial stalk hath he gone by our watch.

HORATIO.
In what particular thought to work I know not;
But in the gross and scope of my opinion,
This bodes some strange eruption to our state.

MARCELLUS.
Good now, sit down, and tell me, he that knows,
Why this same strict and most observant watch
So nightly toils the subject of the land,
And why such daily cast of brazen cannon
And foreign mart for implements of war;
Why such impress of shipwrights, whose sore task
Does not divide the Sunday from the week.
What might be toward, that this sweaty haste
Doth make the night joint-labourer with the day:
Who is't that can inform me?

HORATIO.
That can I;
At least, the whisper goes so. Our last King,
Whose image even but now appear'd to us,
Was, as you know, by Fortinbras of Norway,
Thereto prick'd on by a most emulate pride,
Dar'd to the combat; in which our valiant Hamlet,
For so this side of our known world esteem'd him,
Did slay this Fortinbras; who by a seal'd compact,
Well ratified by law and heraldry,
Did forfeit, with his life, all those his lands
Which he stood seiz'd of, to the conqueror;
Against the which, a moiety competent
Was gaged by our King; which had return'd
To the inheritance of Fortinbras,
Had he been vanquisher; as by the same cov'nant
And carriage of the article design'd,
His fell to Hamlet. Now, sir, young Fortinbras,
Of unimproved mettle, hot and full,
Hath in the skirts of Norway, here and there,
Shark'd up a list of lawless resolutes,
For food and diet, to some enterprise
That hath a stomach in't; which is no other,
As it doth well appear unto our state,
But to recover of us by strong hand
And terms compulsatory, those foresaid lands
So by his father lost. And this, I take it,
Is the main motive of our preparations,
The source of this our watch, and the chief head
Of this post-haste and rummage in the land.

BARNARDO.
I think it be no other but e'en so:
Well may it sort that this portentous figure
Comes armed through our watch so like the King
That was and is the question of these wars.

HORATIO.
A mote it is to trouble the mind's eye.
In the most high and palmy state of Rome,
A little ere the mightiest Julius fell,
The graves stood tenantless and the sheeted dead
Did squeak and gibber in the Roman streets;
As stars with trains of fire and dews of blood,
Disasters in the sun; and the moist star,
Upon whose influence Neptune's empire stands,
Was sick almost to doomsday with eclipse.
And even the like precurse of fierce events,
As harbingers preceding still the fates
And prologue to the omen coming on,
Have heaven and earth together demonstrated
Unto our climatures and countrymen.

Re-enter Ghost.

But, soft, behold! Lo, where it comes again!
I'll cross it, though it blast me. Stay, illusion!
If thou hast any sound, or use of voice,
Speak to me.
If there be any good thing to be done,
That may to thee do ease, and grace to me,
Speak to me.
If thou art privy to thy country's fate,
Which, happily, foreknowing may avoid,
O speak!
Or if thou hast uphoarded in thy life
Extorted treasure in the womb of earth,
For which, they say, you spirits oft walk in death,
Speak of it. Stay, and speak!

The cock crows.

Stop it, Marcellus!

MARCELLUS.
Shall I strike at it with my partisan?

HORATIO.
Do, if it will not stand.

BARNARDO.
'Tis here!

HORATIO.
'Tis here!

Exit Ghost.

MARCELLUS.
'Tis gone!
We do it wrong, being so majestical,
To offer it the show of violence,
For it is as the air, invulnerable,
And our vain blows malicious mockery.

BARNARDO.
It was about to speak, when the cock crew.

HORATIO.
And then it started, like a guilty thing
Upon a fearful summons. I have heard
The cock, that is the trumpet to the morn,
Doth with his lofty and shrill-sounding throat
Awake the god of day; and at his warning,
Whether in sea or fire, in earth or air,
Th'extravagant and erring spirit hies
To his confine. And of the truth herein
This present object made probation.

MARCELLUS.
It faded on the crowing of the cock.
Some say that ever 'gainst that season comes
Wherein our Saviour's birth is celebrated,
The bird of dawning singeth all night long;
And then, they say, no spirit dare stir abroad,
The nights are wholesome, then no planets strike,
No fairy takes, nor witch hath power to charm;
So hallow'd and so gracious is the time.

HORATIO.
So have I heard, and do in part believe it.
But look, the morn in russet mantle clad,
Walks o'er the dew of yon high eastward hill.
Break we our watch up, and by my advice,
Let us impart what we have seen tonight
Unto young Hamlet; for upon my life,
This spirit, dumb to us, will speak to him.
Do you consent we shall acquaint him with it,
As needful in our loves, fitting our duty?

MARCELLUS.
Let's do't, I pray, and I this morning know
Where we shall find him most conveniently.

Exeunt.

It is a truth universally acknowledged, that a single man in possession
of a good fortune must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families, that he is considered as the rightful
property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that
Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she
told me all about it."

Mr. Bennet made no answer.

"Do not you want to know who has taken it?" cried his wife, impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken
by a young man of large fortune from the north of England; that he came
down on Monday in a chaise and four to see the place, and was so much
delighted with it that he agreed with Mr. Morris immediately; that he is
to take possession before Michaelmas, and some of his servants are to be
in the house by the end of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh, single, my dear, to be sure! A single man of large fortune; four or
five thousand a year. What a fine thing for our girls!"

"How so? how can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome? You
must know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design? Nonsense, how can you talk so! But it is very likely that he
may fall in love with one of them, and therefore you must visit him as
soon as he comes."

"I see no occasion for that. You and the girls may go--or you may send
them by themselves, which perhaps will be still better; for as you are
as handsome as any of them, Mr. Bingley might like you the best of the
party."

"My dear, you flatter me. I certainly have had my share of beauty, but
I do not pretend to be anything extraordinary now. When a woman has five
grown-up daughters, she ought to give over thinking of her own beauty."

"In such cases, a woman has not often much beauty to think of."

"But, my dear, you must indeed go and see Mr. Bingley when he comes into
the neighbourhood."

"It is more than I engage for, I assure you."

"But consider your daughters. Only think what an establishment it would
be for one of them. Sir William and Lady Lucas are determined to go,
merely on that account; for in general, you know, they visit no new
comers. Indeed you must go, for it will be impossible for us to visit
him, if you do not."

"You are over scrupulous, surely. I dare say Mr. Bingley will be very
glad to see you; and I will send a few lines by you to assure him of my
hearty consent to his marrying whichever he chooses of the girls--though
I must throw in a good word for my little Lizzy."

"I desire you will do no such thing. Lizzy is not a bit better than the
others: and I am sure she is not half so handsome as Jane, nor half so
good-humoured as Lydia. But you are always giving her the preference."

"They have none of them much to recommend them," replied he: "they are
all silly and ignorant like other girls; but Lizzy has something more of
quickness than her sisters."

"Mr. Bennet, how can you abuse your own children in such a way? You take
delight in vexing me. You have no compassion on my poor nerves."

"You mistake me, my dear. I have a high respect for your nerves. They
are my old friends. I have heard you mention them with consideration
these twenty years at least."

"Ah, you do not know what I suffer."

"But I hope you will get over it, and live to see many young men of four
thousand a year come into the neighbourhood."

"It will be no use to us, if twenty such should come, since you will not
visit them."

"Depend upon it, my dear, that when there are twenty, I will visit them
all."

Mr. Bennet was so odd a mixture of quick parts, sarcastic humour,
reserve, and caprice, that the experience of three-and-twenty years had
been insufficient to make his wife understand his character. Her mind
was less difficult to develope. She was a woman of mean understanding,
little information, and uncertain temper. When she was discontented, she
fancied herself nervous. The business of her life was to get her
daughters married: its solace was visiting and news.

Call me Ishmael. Some years ago--never mind how long precisely--having
little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery part
of the world. It is a way I have of driving off the spleen and
regulating the circulation. Whenever I find myself growing grim about
the mouth; whenever it is a damp, drizzly November in my soul; whenever
I find myself involuntarily pausing before coffin warehouses, and
bringing up the rear of every funeral I meet; and especially whenever
my hypos get such an upper hand of me, that it requires a strong moral
principle to prevent me from deliberately stepping into the street, and
methodically knocking people's hats off--then, I account it high time to
get to sea as soon as I can. This is my substitute for pistol and ball.
With a philosophical flourish Cato throws himself upon his sword; I
quietly take to the ship. There is nothing surprising in this. If they
but knew it, almost all men in their degree, some time or other,
cherish very nearly the same feelings towards the ocean with me.

There now is your insular city of the Manhattoes, belted round by
wharves as Indian isles by coral reefs--commerce surrounds it with her
surf. Right and left, the streets take you waterward. Its extreme
downtown is the battery, where that noble mole is washed by waves, and
cooled by breezes, which a few hours previous were out of sight of
land. Look at the crowds of water-gazers there.

Circumambulate the city of a dreamy Sabbath afternoon. Go from Corlears
Hook to Coenties Slip, and from thence, by Whitehall, northward. What
do you see?--Posted like silent sentinels all around the town, stand
thousands upon thousands of mortal men fixed in ocean reveries. Some
leaning against the spiles; some seated upon the pier-heads; some
looking over the bulwarks of ships from China; some high aloft in the
rigging, as if striving to get a still better seaward peep. But these
are all landsmen; of week days pent up in lath and plaster--tied to
counters, nailed to benches, clinched to desks. How then is this? Are
the green fields gone? What do they here?

But look! here come more crowds, pacing straight for the water, and
seemingly bound for a dive. Strange! Nothing will content them but the
extremest limit of the land; loitering under the shady lee of yonder
warehouses will not suffice. No. They must get just as nigh the water
as they possibly can without falling in. And there they stand--miles of
them--leagues. Inlanders all, they come from lanes and alleys, streets
and avenues--north, east, south, and west. Yet here they all unite. Tell
me, does the magnetic virtue of the needles of the compasses of all
those ships attract them thither?

Once more. Say you are in the country; in some high land of lakes. Take
almost any path you please, and ten to one it carries you down in a
dale, and leaves you there by a pool in the stream. There is magic in
it. Let the most absent-minded of men be plunged in his deepest
reveries--stand that man on his legs, set his feet a-going, and he will
infallibly lead you to water, if water there be in all that region.
Should you ever be athirst in the great American desert, try this
experiment, if your caravan happen to be supplied with a metaphysical
professor. Yes, as every one knows, meditation and water are wedded for
ever.

But here is an artist. He desires to paint you the dreamiest, shadiest,
quietest, most enchanting bit of romantic landscape in all the valley
of the Saco. What is the chief element he employs? There stand his
trees, each with a hollow trunk, as if a hermit and a crucifix were
within; and here sleeps his meadow, and there sleep his cattle; and up
from yonder cottage goes a sleepy smoke. Deep into distant woodlands
winds a mazy way, reaching to overlapping spurs of mountains bathed in
their hill-side blue. But though the picture lies thus tranced, and
though this pine-tree shakes down its sighs like leaves upon this
shepherd's head, yet all were vain, unless the shepherd's eye were
fixed upon the magic stream before him. Go visit the Prairies in June,
when for scores on scores of miles you wade knee-deep among
Tiger-lilies--what is the one charm wanting?--Water--there is not a drop
of water there! Were Niagara but a cataract of sand, would you travel
your thousand miles to see it? Why did the poor poet of Tennessee, upon
suddenly receiving two handfuls of silver, deliberate whether to buy
him a coat, which he sadly needed, or invest his money in a pedestrian
trip to Rockaway Beach? Why is almost every robust healthy boy with a
robust healthy soul in him, at some time or other crazy to go to sea?
Why upon your first voyage as a passenger, did you yourself feel such a
mystical vibration, when first told that you and your ship were now out
of sight of land? Why did the old Persians hold the sea holy? Why did
the Greeks give it a separate deity, and own brother of Jove? Surely
all this is not without meaning. And still deeper the meaning of that
story of Narcissus, who because he could not grasp the tormenting, mild
image he saw in the fountain, plunged into it and was drowned. But that
same image, we ourselves see in all rivers and oceans. It is the image
of the ungraspable phantom of life; and this is the key to it all.

Now, when I say that I am in the habit of going to sea whenever I begin
to grow hazy about the eyes, and begin to be over conscious of my
lungs, I do not mean to have it inferred that I ever go to sea as a
passenger. For to go as a passenger you must needs have a purse, and a
purse is but a rag unless you have something in it. Besides, passengers
get sea-sick--grow quarrelsome--don't sleep of nights--do not enjoy
themselves much, as a general thing;--no, I never go as a passenger;
nor, though I am something of a salt, do I ever go to sea as a
Commodore, or a Captain, or a Cook. I abandon the glory and distinction
of such offices to those who like them. For my part, I abominate all
honorable respectable toils, trials, and tribulations of every kind
whatsoever. It is quite as much as I can do to take care of myself,
without taking care of ships, barques, brigs, schooners, and what not.
And as for going as cook,--though I confess there is considerable glory
in that, a cook being a sort of officer on ship-board--yet, somehow, I
never fancied broiling fowls;--though once broiled, judiciously
buttered, and judgmatically salted and peppered, there is no one who
will speak more respectfully, not to say reverentially, of a broiled
fowl than I will. It is out of the idolatrous dotings of the old
Egyptians upon broiled ibis and roasted river horse, that you see the
mummies of those creatures in their huge bake-houses the pyramids.

No, when I go to sea, I go as a simple sailor, right before the mast,
plumb down into the forecastle, aloft there to the royal mast-head.
True, they rather order me about some, and make me jump from spar to
spar, like a grasshopper in a May meadow. And at first, this sort of
thing is unpleasant enough. It touches one's sense of honor,
particularly if you come of an old established family in the land, the
Van Rensselaers, or Randolphs, or Hardicanutes. And more than all, if
just previous to putting your hand into the tar-pot, you have been
lording it as a country schoolmaster, making the tallest boys stand in
awe of you. The transition is a keen one, I assure you, from a
schoolmaster to a sailor, and requires a strong decoction of Seneca and
the Stoics to enable you to grin and bear it. But even this wears off
in time.

What of it, if some old hunks of a sea-captain orders me to get a broom
and sweep down the decks? What does that indignity amount to, weighed,
I mean, in the scales of the New Testament? Do you think the archangel
Gabriel thinks anything the less of me, because I promptly and
respectfully obey that old hunks in that particular instance? Who ain't
a slave? Tell me that. Well, then, however the old sea-captains may
order me about--however they may thump and punch me about, I have the
satisfaction of knowing that it is all right; that everybody else is
one way or other served in much the same way--either in a physical or
metaphysical point of view, that is; and so the universal thump is
passed round, and all hands should rub each other's shoulder-blades,
and be content.

Again, I always go to sea as a sailor, because they make a point of
paying me for my trouble, whereas they never pay passengers a single
penny that I ever heard of. On the contrary, passengers themselves must
pay. And there is all the difference in the world between paying and
being paid. The act of paying is perhaps the most uncomfortable
infliction that the two orchard thieves entailed upon us. But being
paid,--what will compare with it? The urbane activity with which a man
receives money is really marvellous, considering that we so earnestly
believe money to be the root of all earthly ills, and that on no
account can a monied man enter heaven. Ah! how cheerfully we consign
ourselves to perdition!

Finally, I always go to sea as a sailor, because of the wholesome
exercise and pure air of the fore-castle deck. For as in this world,
head winds are far more prevalent than winds from astern (that is, if
you never violate the Pythagorean maxim), so for the most part the
Commodore on the quarter-deck gets his atmosphere at second hand from
the sailors on the forecastle. He thinks he breathes it first; but not
so. In much the same way do the commonalty lead their leaders in many
other things, at the same time that the leaders little suspect it. But
wherefore it was that after having repeatedly smelt the sea as a
merchant sailor, I should now take it into my head to go on a whaling
voyage; this the invisible police officer of the Fates, who has the
constant surveillance of me, and secretly dogs me, and influences me in
some unaccountable way--he can better answer than any one else. And,
doubtless, my going on this whaling voyage, formed part of the grand
programme of Providence that was drawn up a long time ago. It came in
as a sort of brief interlude and solo between more extensive
performances. I take it that this part of the bill must have run
something like this:

"Grand Contested Election for the Presidency of the United States.
"WHALING VOYAGE BY ONE ISHMAEL. "BLOODY BATTLE IN AFFGHANISTAN."

Though I cannot tell why it was exactly that those stage managers, the
Fates, put me down for this shabby part of a whaling voyage, when
others were set down for magnificent parts in high tragedies, and short
and easy parts in genteel comedies, and jolly parts in farces--though I
cannot tell why this was exactly; yet, now that I recall all the
circumstances, I think I can see a little into the springs and motives
which being cunningly presented to me under various disguises, induced
me to set about performing the part I did, besides cajoling me into the
delusion that it was a choice resulting from my own unbiased freewill
and discriminating judgment.

Chief among these motives was the overwhelming idea of the great whale
himself. Such a portentous and mysterious monster roused all my
curiosity. Then the wild and distant seas where he rolled his island
bulk; the undeliverable, nameless perils of the whale; these, with all
the attending marvels of a thousand Patagonian sights and sounds,
helped to sway me to my wish. With other men, perhaps, such things
would not have been inducements; but as for me, I am tormented with an
everlasting itch for things remote. I love to sail forbidden seas, and
land on barbarous coasts. Not ignoring what is good, I am quick to
perceive a horror, and could still be social with it--would they let
me--since it is but well to be on friendly terms with all the inmates of
the place one lodges in.

By reason of these things, then, the whaling voyage was welcome; the
great flood-gates of the wonder-world swung open, and in the wild
conceits that swayed me to my purpose, two and two there floated into
my inmost soul, endless processions of the whale, and, mid most of them
all, one grand hooded phantom, like a snow hill in the air.

In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon
the face of the deep. And the Spirit of God moved upon the face of the
waters. And God said, Let there be light: and there was light. And God saw the light, that it was good: and God divided the light
from the darkness. And God called the light Day, and the darkness he called Night.
And the evening and the morning were the first day. And God said, Let there be a firmament in the midst of the waters,
and let it divide the waters from the waters. And God made the firmament, and divided the waters which were
under the firmament from the waters which were above the firmament:
and it was so. And God called the firmament Heaven. And the evening and the
morning were the second day. And God said, Let the waters under the heaven be gathered together
unto one place, and let the dry land appear: and it was so. And God called the dry land Earth; and the gathering together of
the waters called he Seas: and God saw that it was good. And God said, Let the earth bring forth grass, the herb yielding
seed, and the fruit tree yielding fruit after his kind, whose seed is
in itself, upon the earth: and it was so. And the earth brought forth grass, and herb yielding seed after
his kind, and the tree yielding fruit, whose seed was in itself, after
his kind: and God saw that it was good. And the evening and the morning were the third day. And God said, Let there be lights in the firmament of the heaven
to divide the day from the night; and let them be for signs, and for
seasons, and for days, and years: And let them be for lights in
the firmament of the heaven to give light upon the earth: and it was
so. And God made two great lights; the greater light to rule the day,
and the lesser light to rule the night: he made the stars also. And God set them in the firmament of the heaven to give light
upon the earth, And to rule over the day and over the night, and
to divide the light from the darkness: and God saw that it was good. And the evening and the morning were the fourth day. And God said, Let the waters bring forth abundantly the moving
creature that hath life, and fowl that may fly above the earth in the
open firmament of heaven. And God created great whales, and every living creature that
moveth, which the waters brought forth abundantly, after their kind,
and every winged fowl after his kind: and God saw that it was good. And God blessed them, saying, Be fruitful, and multiply, and fill
the waters in the seas, and let fowl multiply in the earth. And the evening and the morning were the fifth day. And God said, Let the earth bring forth the living creature after
his kind, cattle, and creeping thing, and beast of the earth after his
kind: and it was so. And God made the beast of the earth after his kind, and cattle
after their kind, and every thing that creepeth upon the earth after
his kind: and God saw that it was good. And God said, Let us make man in our image, after our likeness:
and let them have dominion over the fish of the sea, and over the fowl
of the air, and over the cattle, and over all the earth, and over
every creeping thing that creepeth upon the earth. So God created man in his own image, in the image of God created
he him; male and female created he them. And God blessed them, and God said unto them, Be fruitful, and
multiply, and replenish the earth, and subdue it: and have dominion
over the fish of the sea, and over the fowl of the air, and over every
living thing that moveth upon the earth. And God said, Behold, I have given you every herb bearing seed,
which is upon the face of all the earth, and every tree, in the which
is the fruit of a tree yielding seed; to you it shall be for meat. And to every beast of the earth, and to every fowl of the air,
and to every thing that creepeth upon the earth, wherein there is
life, I have given every green herb for meat: and it was so. And God saw every thing that he had made, and, behold, it was
very good. And the evening and the morning were the sixth day.
"""
