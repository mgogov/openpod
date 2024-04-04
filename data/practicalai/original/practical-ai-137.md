**Daniel Whitenack:** Welcome to another episode of Practical AI. This is Daniel Whitenack. I am a data scientist with SIL International, and I'm joined as always by my co-host, Chris Benson, who is a strategist with Lockheed Martin. How are you doing, Chris?

**Chris Benson:** I am doing very well. How's it going today, Daniel?

**Daniel Whitenack:** It's going good. For listeners who are in the U.S, you'll know that this last week was Memorial Day weekend... Which is good, it was a nice, long weekend, it's a beautiful weather here, but I'm feeling the amount of catch-up that I need to do in this short week. I don't know about yourself...

**Chris Benson:** No, they gave us a holiday, but then the amount of work for that week never changed, so you just have to cram it all in.

**Daniel Whitenack:** It's always the same, right? Yeah, I'm feeling it this week... But you know, lots of good stuff to work on, but I'll sleep well over the weekend, I'm sure, and have a nice Saturday to sleep in, hopefully...

**Chris Benson:** There you go. Recovery time this weekend.

**Daniel Whitenack:** Recovery time this weekend. One thing is there's a lot of news and updates and AI stuff to catch up on.

**Chris Benson:** Indeed.

**Daniel Whitenack:** I know both you and I have probably been feeling that. Lots of things have been happening in the AI world, so today on our episode what we're gonna do is we're just gonna kick it back old school, and as our long-term listeners will know, sometimes Chris and I like to do these episodes where we don't really have a specific guest speaking about a topic, but we kind of talk about a variety of topics that we've seen in the AI ecosystem, that we think are interesting, maybe of note, and discuss them here live. So are you up for that, Chris?

**Chris Benson:** I'm totally up for that. We have fun with these episodes. I enjoy them. This is a little freeform for us.

**Daniel Whitenack:** \[04:10\] Exactly, yeah. And I'm sure it shows to our listeners how ignorant we are in certain cases, but that's okay... Because as everybody who's exploring this space, everybody has their own little niche sphere of knowledge, and it seems like these other people are very knowledgeable in all of AI, but they just have their own little niche of knowledge too, so it's good to sort of dip our toes into different areas... At least it is for me, I think.

**Chris Benson:** That is something that we talked about a little bit in the last episode - for anybody that might have been listening - was just the fact that there's too much stuff for everyone to know all of it, so you kind of have to pick and choose where you're gonna dive in... And that might be something worth talking about today, because you have to kind of make some choices, and it's interesting how the different choices people make affects how a team might come together, and what the team's capabilities are, and stuff. So there's a lot of ramifications here.

**Daniel Whitenack:** Yeah, there's definitely the concept as well of like when you're putting together a data science team or an AI team or something like that -- it's funny we're actually bringing this up, because we had one of these discussions internally even this week, like "Hey, what are the advantages of having a sort of organizationally separate data science and AI unit, versus data science and AI people embedded in various teams throughout the organization?" And also within those specific people, of course, sometimes people sort of want to hire in data science people or AI people to just do the sort of modeling and analysis stuff. And like "Don't worry about the other things, just focus on math stuff and we'll handle the business logic and pushing that out into products and all of that stuff."

In other cases, people very much like to -- you know, there's even this term now "full stack data science", where it's like "Hey, I'm gonna take care of doing this analysis piece, but I'm also gonna figure out how to integrate it into a product." So there's just so many different parameters there in terms of how people put their teams together... And it makes it hard for a new person to figure out where they fit in that, I think.

**Chris Benson:** It sure does. Not only that, but it doesn't stand by itself, in that AI, if you're thinking of AI as deep learning, which we tend to in these last few years, and it has to be -- you know, you have the algorithms, you have the need to deploy it out, to be useful to your user community, it integrates in with other software, it integrates in with the microservices and the APIs... So you have a huge area of target surface of things that have to get done, and I have never seen two organizations do it the same way, in the organizations that I've worked with. It's like, everyone has their own way of doing it, which means there's not a normalized or standardized way to achieve that.

**Daniel Whitenack:** Yeah. And of course, this is something -- because we have a podcast or we're interacting with a lot of different groups, something that I'm asked a lot is "Hey, I'm wanting to get into this field. Where do I need to focus?" And oftentimes my response to that is "Well, let me ask you a question... Are you more interested in this sort of more engineering slant, or are you just wanting to do researchy type things and analysis and that sort of thing?" Because those two could take a very different path, and probably if you're focusing on one, you're not gonna be well-prepared for the other one. It's very rare that there's someone that operates in both of those spaces.

**Chris Benson:** Yeah, totally. It's a super-hard challenge to figure out for someone new into it, because I always ask people what they're already passionate about, or I ask them what kinds of things are they reading, and watching videos on, that has captured their imagination... And then start building on that. And you can ask five different people and get five very, very different answers, which isn't necessarily a bad thing. If they're each gravitating toward different areas that have to be addressed, then that's the makings for a team right there.

**Daniel Whitenack:** \[08:04\] Yeah. Maybe one thing I could suggest to promote my own biased opinion - I was thinking about this recently... I'm kind of involved in some respect over at Purdue University; sometimes some of their students help my organization with some projects, like practicum type projects, or I help their students in working on projects, or occasionally I teach some lectures... And I was putting together some material for a course; this time around, my thought process was really like "For these students, if I was hiring them in, what would be the set of things I would want them to go through or be able to go through that would really give me confidence in their abilities to perform on my data science or AI team?"

So the thing -- and you could disagree with this; as you say, every team is different. But what I did was I proposed a certain problem, the problem was question answering; so a piece of text comes in, there's a question, you have a model that actually extracts the answer out of the text... And this is a well-known problem, and there's models out there that do this... Because it's a short time that we have the students together, I was basically saying "Hey, this model exists, so figure out how to use it for inference, and then I would like to see three different manifestations of that model. One that I can have a REST API and send a request and get a response, one where there's a user interface where I type in my question and I type in my passage and get the answer, and one where I can use it in a batch way." So if I wanna answer -- I have a dataset of 20,000, 50,000, 100,000 questions and passage, and I wanna answer them all at once, what's a way that I can do that in a batch sense?

And I think thinking through those different manifestations of how a model could operate was really -- I think it was actually helpful for me in terms of thinking through how to structure the material, but also I thought it was a good way to think about things and maybe something I'd share on the podcast, because as people are getting into things, they wanna sort of show a project or something... And I would encourage people, you know, maybe it's worth -- you create a model, but then you create these sort of different manifestations of the model in how it actually interacts with users or the end use case, something like that.

**Chris Benson:** Yeah. So let me ask you a question about that... Was the model itself fundamentally designed differently, or was it more of a software input/output thing?

**Daniel Whitenack:** The model itself was basically the same in the three cases, although in a more batched sense--

**Chris Benson:** I had a feeling you were going there...

**Daniel Whitenack:** ...one thing of course you can do is operate on -- you know, instead of pushing one input into the model and getting one \[unintelligible 00:10:45.14\] of course, you can push multiple in... And then you can also parallelize across the dataset and all those sorts of things. There is maybe a different inference pattern. The model architecture itself wouldn't be fundamentally different.

I think if there was more time, I would have loved to add an upfront component which would have been like take a pre-trained model, fine-tune that model with just a little bit of data for a slightly different task or a slightly different domain, and then sort of make it available in these three different ways.

That's almost a sort of bootcamp in and of itself... If you can go through that process, I think you're at least well set up to talk the language and understand how people go about their AI and ML development. And I'm sure there's people out there that disagree with me.

**Chris Benson:** Yeah, I would argue that that's almost more of an engineering thing than it is truly a modeling thing... Because your model is not fundamentally changing; it's just how you're choosing to deploy it and integrate it in with the larger software environment that you're in.

**Daniel Whitenack:** \[11:53\] Yeah, and I would probably go more -- I'm naturally more on that engineering side probably in the way that I operate than I am in that sort of pure research or creating new model architecture side of things. So if someone wants to go and be the next team developing the next cool model that is fundamentally different, and they spend five years on that to develop that model and really prove it out and probe all of the implications and all of that, that is also really needed. But it's a different skillset around research and experimentation.

**Chris Benson:** You raise a really interesting -- there's a dimension to this I'd like to draw, and tell me if you have a different sensation about this. As I'm looking at the deep learning world over the last year or two compared to the years prior to that, it seems like we're seeing lots of completely new architectures coming through the late teens especially - early and into the late teens - and then we kind of got into the twenties and we're really in an age of implementation.

I don't know if that's because I'm in my bubble and I'm seeing tons and tons and tons of implementation but not a whole lot of truly revolutionary new ideas coming to market... What is your impression? Am I trapped in my little bubble, or do you think that we're seeing more implementation than we are research right now, because so many more people are getting into deep learning?

**Daniel Whitenack:** It's a good question. I think part of it is maybe the maturity of the field, in the sense that in maybe previous years everything seemed a little bit more new because nothing was really normalized yet... It was all this sort of storm of ambiguity and new things. Whereas there's really developed workflows around transfer learning, and even running really large-scale jobs on multiple GPUs and other things; people have developed workflows around that. So maybe those pieces of it seem less new, but I don't know... I don't know if I would say that the progression of new models and architectures has slowed or increased. I don't know. There's definitely still some of that going on, but the perception may be different.

**Chris Benson:** It feels evolutionary at the moment to me... Because I was really wondering this, knowing that we were gonna have this conversation among ourselves, I was coming into it thinking about it and going "When's the last time I saw a completely new architecture sweep through the market and really take hold?" Back when we were looking at the \[unintelligible 00:14:28.19\] convolutional neural networks, and then we hit the age of NLP really hitting its stride, followed immediately by another big jump with transformers, and then we've seen reinforcement learning have some jumps in there as well... And I was thinking "Hm... When's the last time one of those big jumps leapt out at us?" Can you think of what that might have been? What would you think of as the most recent big jump forward?

**Daniel Whitenack:** I don't know... I'm also pretty stuck in my little NLP bubble maybe... Obviously, transformers and large-scale language models have been the thing...

**Chris Benson:** Your bread and butter, yeah. Definitely.

**Daniel Whitenack:** ...that have really pushed the envelope forward... But that really has happened over the past few years. There is one thing I'd like to discuss in a bit, that I think we can decide if it's new or not, in that sense of new.

**Chris Benson:** Okay...

**Daniel Whitenack:** It's new, but yeah... I don't know, I definitely think that there's more people getting into the field, and because of the tooling and the maturity of the tooling, there's more people that are able to do very advanced and sophisticated things, whereas maybe that was restricted to a very elite before.

**Break:** \[15:39\]

**Daniel Whitenack:** Chris, you were kind of bringing up this topic of new architectures, and what is a big leap in terms of new models, and what is building on the shoulders of things before... And to some degree, everything's building on things from before, but one of the things that I've really enjoyed watching and have seen a lot of discussion on Twitter and in the circles that I'm in is this new model from Facebook, wav2vec-u. I don't know if you've seen this; it's a pretty interesting model. It's a speech recognition model, and the interesting thing -- well, I guess it sorts of stands on the shoulders of wav2vec, which was the previous model... But it is very interesting in that it is unsupervised. That's what the u stands for.

**Chris Benson:** Okay...

**Daniel Whitenack:** And I think what maybe shocked people or what got them really thinking is that this wav2vec-u essentially learns how to create transcriptions of audio or speech using purely unpaired data or untranscribed data. So just putting audio in. Which is kind of a weird thing, if you think about it. Have you seen this, or has it crossed your paths at all?

**Chris Benson:** I had seen the name, but I hadn't dug into it. I'm looking at it while we're talking about it... It came out a couple of weeks ago, so it's still very new, if I'm reading this correctly...

**Daniel Whitenack:** It's still fresh... Of course, speech recognition is very much a problem -- or often the problem with speech recognition is the fact that you don't' have a lot of data. We've talked to people like Jeff Adams or others on this podcast who worked in speech technology for some long time, and a lot of times the problem there is you need a good amount of high-quality transcribed audio; we're talking about thousands and thousands of hours of transcribed audio... And you need it pretty high quality, you need a diversity maybe of speakers, you need a diversity of accents... And gathering and curating all that data is incredibly labor-intensive.

So this model basically operates in a slightly different way than previous models in that what it's trying to do is take in unlabeled speech audio (just audio) and generate the phonemes corresponding to that audio. Phonemes is a word that I -- you know, I'm not a linguist, so I learned this after joining with SIL, and they started talking about phones, and I was confused about -- cell phones? What type of phones are you talking about?" But phonemes or phones - here we're talking about actual sounds. So phones are the sounds corresponding to the audio. And what you can do is you can actually try to create a model that takes in speech, and generate phones (or phonemes) from that. Then you can take those phonemes and map them to text. And it's that sort of generating from the audio to the phones that they did in an unsupervised way. And the way that they did this was actually in a sort of generative-adversarial way. So they had audio coming in one stream through a generator model, that tried to generate the phones corresponding to that unlabeled data... And then over here they had actually unlabeled, phonemized text.

\[20:07\] So they took text that was already -- it was good text; it was text that they knew was in the language that they were working with, they converted that to phones, and then they tried to see if a discriminator could tell the difference between these sort of "good phones" that were actually from actual text, versus the phones that were generated from the generator model.

So it's interesting that they're working in two ways - they're working in this phone space, but then they're also using this idea of generative adversarial networks to allow them to solve this problem in an unsupervised way. So I don't know if this fits into -- you know, after talking through it, if this fits into fundamentally new/different, or just incremental... I don't know.

**Chris Benson:** I would say -- I don't know that it's a whole new category, but it's certainly an interesting step forward, and it kind of expands the natural language processing and generative adversarial network world together. I'm kind of curious, could you talk about a use case where you could see this being applied, just to give context for it?

**Daniel Whitenack:** Yeah, good question. Of course, this also stood out to me because I work with a lot of local languages, and oftentimes you either have a very small amount of transcribed audio for a language in one of these lower resource language, or you may have none. But oftentimes in language survey and documentation and language preservation efforts you do collect audio of the language, right? It's just not transcribed. And in some cases maybe it's actually never been written down. There's many languages out there, people might be surprised, that are still just oral languages. They've never even been written down in any type of script.

So I think that although this might not be applied out of the box in that scenario, I think it works us much closer to where people working with these sort of endangered languages and extremely low-resource languages, who are able to gather audio of the language speakers still, they may have an advantage in terms of being able to actually work towards transcribing that audio more quickly than they would have before.

**Chris Benson:** Fair enough. I'm really fascinated by the number of use cases -- you know, we've just seen so many orders of magnitude exploding outward the last 2-3 year in terms of where... You know, when we started this podcast - and we are closing in now on three years that we have been doing this, and the landscape...

**Daniel Whitenack:** That's kind of crazy.

**Chris Benson:** It is a little bit crazy, I agree. The landscape since we started has changed dramatically in terms of how widespread these technologies, and specifically these models being deployed has been. It was still a little bit unusual to find use cases in the marketplace where you had deployed deep learning models that were actually in production. They were kind of here and there, whereas now it is everywhere. And really, going back to some conversations that we had some time back, we are seeing it very much just being part of software development. The idea of model-based engineering is just what engineering has become. It's not a separate thing anymore, it's not a call-out, and I think it is unlikely these days to build any significant software system without any thought into model deployment as a service or microservice. They can be utilized in that... Which brings us back to all of those folks out there who may either be trying to create a career in AI, or deep learning, or however you wanna call it, or even software, and trying to figure out where exactly they fit in, and how they should approach, as well as a lot of organizations. So now that the cost has been driven down so substantially from the early days, when you -- you know, very early, when the widespread, large things were being done by the big-name companies, but now it's available to just about anyone, and all the major cloud providers have full tooling sets that cover not only the model creation and the model deployment, but the entire software stack with it integrated.

\[24:14\] So if you have somebody coming out of college right now, and maybe they have a computer science degree, maybe they don't, maybe they're in another area, but this has captured their imagination, how would they help their new employer get into this, in what different areas? If you were coming in -- it's been a while since we talked about it from a complete newbie standpoint. I'm just curious, how has that changed now in the last couple of years?

**Daniel Whitenack:** So when you say that, do you mean like I am a data or AI person coming into a company that's not doing AI things, how do I go about creating that transformation? Or are you coming at it more from "I am coming out of college, and I want to step into this world of AI" type stuff?

**Chris Benson:** I think we should divide them... Let's hit both questions for a few minutes. I think it's worth asking, because you and I have been doing this for such a long time, and it feels like we've been there and done that many times... But in my day job, outside of the podcast, I'm still having people come to me on a regular, day-to-day basis and asking for mentorship, and how do I do this, and it makes me realize how many people are still trying to find a path into it. That happens a lot.

**Daniel Whitenack:** Yeah. I think there's actually -- and maybe we can go in terms of learning resources for those actually trying to get into the field as an AI person here in a second... But I think that the first of those that you mentioned, about how do you seed or spawn AI work within an organization that's maybe not operating in that area currently - I think that's actually a very challenging problem. I would recommend -- I don't know if you remember... We were talking about how long the podcast has been going on... But way, way, way back at the beginning, in the first few episodes - I forget which of the first few episodes - we had \[unintelligible 00:26:00.25\] on the show...

**Chris Benson:** Oh, yeah. I do remember that.

**Daniel Whitenack:** Yeah... He's sort of an expert in -- I mean, I know he's an expert in this area; I don't know how he would describe himself, but how I sort of think of him is really, really having amazing expertise in behavioral economics of creating this sort of change within an organization. He has a book called Cracking the Data Code, so that's maybe a resource out there for those looking to create more of that organizational change.

He has a formula that he follows, but I think that part of that really comes down to making sure that you have and you develop more and more empathy for the people in your organization and the sort of pain points that they're feeling, the more that they can feel truly listened to, and that you understand their pain points and why they go about the way that they're solving those problems now, and the challenges that they're having, the better change you have of coming beside them and bringing them in very early into the process, and helping them shape what a solution looks like...

Because once you get through a first cycle of this, and if it is successful, other people will sort of start asking, and momentum will build... But if you get that wrong the first time and you develop a really bad taste in people's mouth for AI/predictive things, then that's almost impossible to overcome, I think.

**Chris Benson:** That's a really good point, and that was a really good answer to how to address that. One of the things that I'm seeing also is organizations trying to figure out how to -- kind of what we talked about in the very beginning a little bit, but how to address AI... Do they do it as a separate organization? Do they integrate it in with their data science, or their software teams? And that can vary based on the domain expertise they have, what their employees are, what kind of business they're in, and also what technologies they're focused on.

\[28:03\] I think early on I saw a lot of AI being broken out more and more as its own thing, and I think the people that drove that were very savvy on internal marketing and getting budget for that... And certainly, as we've seen it normalize over time a little bit, you're starting to see it roll back into more of the data science and DevOps side of things as well, since it's just another type... You know, divide up your model types into -- you know, some happen to use GPUs and some don't. That seems a very artificial thing.

**Daniel Whitenack:** Yeah... And I don't think there's any perfect solution here, because there's gonna be challenges either way. If you try to centralize your data science and AI team within an organization, the challenges you're gonna face are those where you have to be very intentional about having those people reach out and make sure that they connect with the end user, understand their pain points, understand their challenges, and that they really have the tentacles out in the organization... Whereas if you have everybody distributed everywhere, you're probably gonna deal with more duplication type issues, duplication of work, how are they gonna share resources, maybe it's more efficient to have centralized infrastructure, but now it's more complicated, because people are in all of these different organizational units, and all of that sort of stuff. So I don't think any one particular model is perfect.

**Break:** \[29:31\]

**Daniel Whitenack:** Well, as always on these Fully Connected shows, I think that our conversation has led us exactly to where we often like to go in these types of episodes, and that's to learning resources for people; maybe new ones that have come out, or things that have been updated... And we are talking about two different scenarios. One where the challenge is having an organization be transformed into one that uses AI or data science type techniques, and the other where maybe you're an individual and you're trying to break into an organization that's already using AI. Maybe on that latter point - I ran across something that I haven't gone through totally, but I very much want to, and I've seen a bit of buzz around... This is this new book that is called Meta Learning. Have you seen this?

**Chris Benson:** I haven't, no.

**Daniel Whitenack:** Okay. Originally, when I saw the book, I thought "Oh, cool. It's a meta learning book", and of course, that's a bit of a loaded term in the deep learning, AI type world, because meta learning -- we even have an episode with... I'm not sure -- well, I know we talked to \[unintelligible 00:31:32.25\] I think we talked to a couple of people about meta learning...

**Chris Benson:** Yeah.

**Daniel Whitenack:** ...where this is really learning to learn. So you have some method that maybe learns an architecture, or learns an optimization technique, or something like that... You're learning to learn in some way - well, that's not the kind of... Well, maybe that's part of the book, I don't know. But this sort of learning to learn book is really about how to learn deep learning. So it is meta learning. It's how to learn deep learning.

**Chris Benson:** \[32:04\] It's the human learning, not the model learning in this case.

**Daniel Whitenack:** Yeah, the human learning, not a model learning how to do learning, but a human learning how to do deep learning. It's a nice title, "How To Learn Deep Learning And Thrive In The Digital World." And sorry if I get the name wrong, Radek Osmulski. It looks like a really nice book, and I think the focus here - he talks about "I learned to program and do deep learning using online resources. Most of my income over the last two years has come from deep learning roles. How did that happen?" And he basically lays it out, how did he go from learning deep learning using online resources, and now his career is doing deep learning type things.

**Chris Benson:** I think both of us ended up in that same path. Most of our learning has come from self-learning.

**Daniel Whitenack:** I mean, certainly I'm grateful for the math foundational things in my past, and certain programming things, experience with doing experiments in science, and that sort of thing... But physics isn't exactly deep learning, and I didn't really think about machine learning until I was in industry. And maybe you can remind people about your status prior to working in AI...

**Chris Benson:** Well, I was doing lots of software for years, but actually, even that would be the same thing, in that I learned how to do software fairly early in my career, and my degree is actually in finance. So it has nothing to do directly with any of these things that we have been talking about, whether it be software, or deep learning, or the things that come next.

So I think the beauty of a book like this, as I'm looking through the web page about it, is that this is really the core about how people build careers now. If you are thinking about how to get there, you have to build in constant, life-long learning into your process, and if you don't do that, you will fall behind. And I meet people all the time that are doing both. I meet people that are constantly learning -- I know you and I are definitely two examples of that. I have some other good friends, good colleagues that are constantly learning as things are moving forward, and figuring out where and how they're going to dig into those ideas incrementally as they go... And then you see people that don't do that, and they do fall behind.

And as things are moving faster and faster forward, the ability to thrive in a digital world, going back to the book's subtitle, is really crucial to having it. Because anything you're learning right now in more of a formal context, if you're coming out of school for instance, a lot of that will be obsolete. The basic math will be there, but the algorithms are gonna change, how you're achieving problem solutions will change, and so you really have to be able to do that. I think that is a fundamental for being a digitally literate person in the job world these days.

**Daniel Whitenack:** Yeah, and also a lot of what I've seen is maybe difficulty in parsing through -- it's kind of a good thing and a bad thing. There's so much available online that you can use for your learning, but there's also so much available online --

**Chris Benson:** For free or very inexpensively, yeah.

**Daniel Whitenack:** So how do you figure out what to focus on when there's so much you could focus on, and in what order? That's often the missing piece. I feel like something like this - and again, I'm expecting the content of this book at least goes into a little bit of this, although I haven't explored it in detail, but it sounds like from the purpose of this that part of it is kind of putting some framing around the resources that are online, and helping people understand the different things that they might need to explore if they want to have this sort of role, or be part of this kind of world... And that sort of framing can be almost the missing piece. Because someone published a course over here on this topic, and someone publishes a course in this topic, and then there's a thousand courses... And there's a route through those courses that makes sense, but if you don't know anything about the topic in the first place, it's hard to determine what that route is. And then you sort of get all confused, and maybe discouraged.

**Chris Benson:** \[36:10\] Yeah, I think a really good tactic - which actually comes straight out of the software world, is the concept of scratching your own itch... It's that if you wanna get into deep learning and into AI over time as it continues to evolve, having a personally significant use case, something that you wanna get done... And it doesn't have to be a day job thing; it can be something you're doing on your own, it could be doing something around the house... There's an infinite number of possibilities, and we've talked about these off and on over the three years of the show so far at various times, in terms of things that we wanted to do. And having a use case and figuring out a path through that - and there can be an infinite number of paths through that; if your fascination is "How do I create the models?", then you spend your time and learning resources on "What are the possibilities?"

Somebody else may be a lot more interested in the engineering side of it, and they're really just keen on taking advantage of transfer learning, and finding a model that's already doing very similar to what you want, if not exactly what you want, and basically getting that into your context and deploying it out there. And those are very, very different paths that you might take to achieve essentially the same thing. But doing that also helps you find your own personal golden path on how you're gonna come into this AI world and be able to be productive and passionate about what you're doing.

So really scratching your own itch, finding the things you wanna do... If it seems a little intimidating, just try to dig in a little bit and try to figure out what those parts would be before you go "Oh, this is too much", and break it down, and then continue to break each piece down and put them in the right order, and get there. I find that I have a habit of -- the things that I've gravitate toward have usually been because of that, because there was a personal fire I had going on that particular thing.

**Daniel Whitenack:** Yeah. The other interesting element of this finding your path -- and I don't know that this is explicitly detailed anywhere; maybe this would be a good resource that could be out there, but depending on where you wanna end up in your career and who you're wanting to work with, the sort of stack that you're working with - it can be very different, as we've talked about... And certain of these courses online assume a single lane, because it's all from one perspective in terms of the tech stack and the tooling as well. I was looking up for some MLOps resources recently...

**Chris Benson:** And by the way, for people who aren't familiar with that, MLOps is kind of the machine learning version of DevOps. Development operations. Getting your stuff out there so people can use it, in a reproducible and standardized way. Keep going; I just wanted to throw that out for those who weren't used to it.

**Daniel Whitenack:** Yup, very good. Yeah, so I was looking for some resources on this at different courses out there that teach MLOps. And all of them that I was looking at seemed really good. For example, I was looking at the DeepLearning.ai new MLOps course. I was sort of looking through there, and I think partly because I've been exposed to some of this stuff, I kind of deduced - and partly from who's teaching it - you can sort of deduce "Hey, this is a very TensorFlow Extended-specific pipeline of tooling." They're working with this framework called TensorFlow Extended, in terms of how they're doing both model serialization, optimization, deployment; sort of from end to end, this is the world that they're living in... Which makes sense for a lot of companies, but there's a lot of companies that have drastically different opinions on that sort of pipeline.

\[39:51\] Another one I was looking at was from Elvis Saravia. He publishes a lot of great content, and he has a site called Machine Learning Ops, a collection of resources on how to facilitate machine learning ops with GitHub. This is all about "Okay, I have code that does machine learning stuff. How can I automate the deployment, and manage testing, and all those things with things like GitHub Actions", which is GitHub's sort of continuous integration/continuous deployment (CI/CD), and how to version things properly, and all that stuff.

But then there's another sort of world where MLOps -- like, if you're in a certain organization, they've bought into a platform, like Domino Data Lab, Data Robot, these sort of things that handle a lot of those things as well, but at an enterprise level... So yeah, there's just these different -- I don't know how to help people navigate through things like that. Because if they choose any one of those courses, that's good content. They're gonna learn a lot. But then if they walk into a company that's doing the other thing, it's not that it's not relevant, but a lot of it is not going to be the same.

**Chris Benson:** Yeah, there's that dichotomy that you are almost going to be required to buy in or invest in some sort of particular flow, an environment, if you will. An ecosystem. You have TensorFlow and PyTorch as two of the giant ecosystems, and there are others as well. Recently we were talking about Apache on the show. So at some point you have to say "Okay, I have to pick something", and you're gonna pick the one that either your company is doing ,or that appeals to you, or you've read an article, or whatever... But it really helps to try to abstract back out a little bit about what it is that those things are trying to do, so you don't lock yourself in.

And at some point, you need to go look at some of the other ecosystems and try to map them to what you've just learned in that first one. That way, you have the ability to understand how you're doing MLOps that's not limited to a single ecosystem or community and how they're doing it.

**Daniel Whitenack:** Yeah, I think that's a great point. I hope that this discussion, this sort of rambling discussion has been useful for people that are trying to navigate this road. We'll provide links to all the things that we talked about in the show notes. We'd really encourage people to join our Slack community by going to Changelog.com/community, where you can share some of your favorite resources and talk about how you've been learning deep learning and other things online.

That recording went pretty quick, Chris...

**Chris Benson:** Yeah, it was a fast one.

**Daniel Whitenack:** Yeah. I feel like there's a lot to talk about and catch up on, but...

**Chris Benson:** Well, we'll have more Fully Connected shows coming up then.

**Daniel Whitenack:** More in the future, yeah. Thanks for the discussion, it was fun.

**Chris Benson:** Thanks a lot, Daniel.

**Daniel Whitenack:** Bye.