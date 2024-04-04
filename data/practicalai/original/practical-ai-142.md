**Daniel Whitenack:** Welcome to another Fully Connected episode, where Chris and I keep you fully connected with everything that's happening in the AI community. We'll take some time to discuss the latest AI news and we'll dig into some learning resources to help you level up your machine learning game. I'm Daniel Whitenack, a data scientist with SIL International, and I'm joined as always by my co-host, Chris Benson, who is a strategist at Lockheed Martin. How are you doing, Chris?

**Chris Benson:** I'm doing very well, Daniel. How's it going?

**Daniel Whitenack:** It's going great. I hear almost congrats are in order. You're about to be an official pilot with your pilot's license, is that right

**Chris Benson:** I actually got the license last Sunday.

**Daniel Whitenack:** Okay. So you are actually official.

**Chris Benson:** Frighteningly, I am a licensed pilot. And I know that has nothing to do with AI, but I appreciate the congrats, so...

**Daniel Whitenack:** I'm sure there's some type of AI system that images the Earth, or airplanes, or manages air traffic control that will find you, at some point.

**Chris Benson:** At some point yes, something like that might exist. And you know, I do work for an aerospace company too, so aside from what I'm doing, there might be things like that out there.

**Daniel Whitenack:** \[04:10\] I'm trying to remember the exact conversation we had, but there was a conversation we had where we talked about this - I think it's persistent surveillance that is being promoted, I think, by at least one company, where essentially you just record video of everything all the time, but in low-quality... Low enough quality to where you can't personally identify all the people walking around and that sort of thing, but high enough quality to where if something goes down, then you can sort of back-track and figure out what happened. I don't know if that's happening above my head now... I don't think Lafayette, Indiana is maybe the highest priority target for that sort of system, but...

**Chris Benson:** If I knew about something like that, I couldn't talk about it...

**Daniel Whitenack:** Oh yeah, of course.

**Chris Benson:** Except -- wait a minute... Except my wife is English, so we spend a fair amount of time in the U.K, and they do have cameras just kind of all over the place there.

**Daniel Whitenack:** Sure, sure.

**Chris Benson:** You cannot walk through London without being filmed all over the place. So that's just normal life. In the U.S. that's becoming more and more normal life as well, because -- I don't know about you, but everybody has a Ring or a Nest on their door at this point.

**Daniel Whitenack:** Yeah, definitely a lot. And we have cameras at my wife's business, all around the business, and all sorts of that stuff. The network video recorders now - you can record so much for so cheap, because you just load them with some state drives, or even spinning disks, and you can get a lot of data.

**Chris Benson:** Yeah. It's kind of funny, on our Nest Cam we \[unintelligible 00:05:46.06\] and we are talking a little bit about AI-related stuff, but just on our Nest Cam we found an Amazon driver who decided to put a package into our mailbox, which isn't legal, and stuff... But I was kind of amazed - in an age where not only are there cameras pretty much everywhere now, but now there's all of this automation. There's tons of deep learning analysis, super-cheap, deployed all over the place. Folks, be careful what you do. There's somebody watching.

**Daniel Whitenack:** The building that my wife's business is in, and then we own a second building, which her business will expand into... And right now, there's not a lot in there, and there's no internet over there... So as a quasi-security camera solution now they have deer cams, or wildlife cams that people out out in the--

**Chris Benson:** I have tons of those.

**Daniel Whitenack:** Yeah, that people put out in the forest... But now they have the ones where -- like, when they see an event, they work off the cell signal, and they can ping your phone... But also, they have built-in pre-trained models for different animals - deer, or turkeys, or whatever. I don't think there was a people one in the one that I got... Hopefully not that many people are hunting for people, but -- I mean, we're not using them for hunting, we're using them for this purpose... But yeah.

**Chris Benson:** You're raising a really good point, and that is AI is just everywhere these days. These convolutional models are so cheap to deploy... And aside from us talking about AI from the AI industry perspective, the animal protection non-profit that my wife and I run - we have tons of those cameras, and yes, you are seeing these pre-trained models in these common tools, and these consumers have no idea.

**Daniel Whitenack:** Well, it's a lot of data to sift through.

**Chris Benson:** It is. That is starting to happen right now, and I've gotta tell you - I would have thought that's about as far from AI as I could possibly be... We're seeing models now turn up in the most unlikely of places, it seems.

**Daniel Whitenack:** \[07:51\] I guess we can turn to what we had talked about discussing today, which I think is a really interesting topic. It's something I've been thinking a lot about recently... Is building a data team in a company. There's a really interesting article I saw last week (very recent) from Erik Bernhardsson about "Building a data team in a mid-stage startup: a short story." The format of the article is quite interesting, it's almost like a parable. So he's sort of taken a bunch of his experiences over the years building data teams and wrapped them up in this sort of parable about a data scientist coming into a new company, charged with building a data team, and the experiences that such a person might encounter, which I think is quite intriguing and refreshing to read.

**Chris Benson:** You had shared the article with me before the episode, and he puts it into the reader's perspective there... You know, you're the one going through it. "You notice a lot of code starts to...", that kind of thing. It was an interesting perspective shift, because it puts it onto the reader. I liked it.

**Daniel Whitenack:** It definitely got me thinking about a lot of the experiences I've had in the past, and I don't know about you -- you've been at several places, I'm sure...

**Chris Benson:** Both big and small.

**Daniel Whitenack:** ...so as part of those experiences I'm sure you've been charged at one time or another with maybe not always starting the first data team at a company, but building a data team at a company. Is that part of your experiences in the past?

**Chris Benson:** It is indeed, and actually I have started the data team more than once... Just mainly because I've been working for a long time, and a while back companies didn't even have data teams; they didn't exist. And ironically, for a while you just had a DBA, a database administrator, and that person was expected to do anything that had anything to do with data for a long time. The idea of data teams is a relatively recent thing in terms of mainstream... And when I say that, meaning a number of years now. But it hasn't been decades.

**Daniel Whitenack:** Yeah. And by data team, it's probably worth us discussing even that term. In my mind, I have this vision of a team of mostly data sciency people, but maybe with some kind of towards the side of infrastructure more, and some maybe more towards the side of experimentation, and research or prototyping type of people. When you think of a data team, what comes to your mind? I'm curious.

**Chris Benson:** Usually, it's a little bit different. And it's still very unique, there's not a standard concept for what a data team is... And I've had some missed fires in that. There are different roles, and those roles are as distinct as the software development world has become, with developers and different dev sec ops people, or DevOps people... It's maturing is what's really happening. It's been maturing rapidly, because this AI thing, this deep learning taking over the world has been happening so fast. Before this era, not a lot of companies had dedicated data teams. So we're still in that process of figuring it out.

We have lots of them at my employer, and they don't all -- even in the organization, they don't all think of themselves in the same construct. You can ask different data teams what is a data team and they'll give you different answers.

**Daniel Whitenack:** Yeah. One anecdote that I actually was discussing with someone recently, a friend from college - which I won't share the details, because I didn't get his permission...

**Chris Benson:** Sure.

**Daniel Whitenack:** Basically, the story was he has an engineering background, he's working in the industry, and he had an opportunity within a company to -- because they knew he had some coding skills, and some modeling skills, and that sort of thing, they basically wanted him to sort of become a data scientist within the company, because he had a lot of the industry knowledge, and they knew he was sort of gifted on the coding and programming and modeling side... So they said "Hey, why don't you start our in-house data science team?"

\[12:04\] And I think what he's found is there are people throughout the company that are doing data sciency things; they're not really coordinated maybe well yet, there's not a lot of MLOps and good operational and deployment strategies yet... So that's a lot of what he's parsing through, is what are those best practices. And they're not giving him a full data engineering team to solve both of those things for him; they're saying "Hey, you figure out a way to do it with these people that have been identified as data scientists or data analysts throughout the company... Figure out how to deploy models such that people can use them, but you're gonna have to figure it out, and we don't really have a lot of pre-built infrastructure for that." So yeah, it's a big challenge.

**Chris Benson:** That actually happens at companies large and small. I've seen that in both. And it's interesting as you try to learn how people have moved in. I think a lot of early people have come in - which certainly includes me, and I think to some degree includes you - from the software development side, with less experience on the pure data sciency world. And there's a ramp-up for people that are moving into those kind of roles... Because just because you've coded doesn't mean that you understand statistics deeply and you understand the various mathematical constructs that you need to apply to this... So I know for me -- some of which I had in school a long time ago, but there was definitely a ramp-up for me to be able to be productive, especially 4, 5, 6 years ago, when before things were quite so popular and when individuals were kind of doing everything end-to-end.

**Daniel Whitenack:** Yeah. One of the points that was brought out in this article is that sometimes companies hire in or promote people to do "AI or whatever", and figure out where it should be done, and come to find out maybe the immediate needs aren't that machine learning or AI stuff, or maybe you can't even get to those yet, but it's sort of equal type things, or people that say "I wish I could figure out this number or this metric", and it's totally accessible, they just can't translate those words into either SQL, or scripting, or whatever it takes to pull, extract that data out from its various sources and get that in front of them. That's one of the things that is needed most often first.

**Chris Benson:** I created a little bit of a nightmare in that area myself. I was at a previous employer, I was hired in -- it was a large company that everyone's heard of, I was hired in to create the first AI team in that organization. And it was early enough to where -- I had come into that from generally smaller operations, and I made the mistake of hiring a team of data scientists. And in this case, they were true data scientists, but I ended up making assumptions about what those capabilities those individuals had. They were all probably better at the mathematics of deep learning than I was.

I had self-taught and self-studied and I could hold my own, but that's what all of their formal education had been. They had all recently, for the most part, come out of college, out of university... But then when we got to the point where we needed to do DevOps and deployment and all that, there was absolutely no understanding. Everything from SQL to what is a container, all these things. It was an interesting ramp-up experience and I had to make some course corrections late on after hiring several people in that capacity; I had to specifically recognize that there were other skills that I had not addressed at all, and go hire people for those \[unintelligible 00:15:59.26\]

**Break**: \[16:06\]

**Daniel Whitenack:** Chris, you brought up a really interesting point which crosses over into the hiring side of things, which is -- there's a few scenarios that could happen. You could hire in machine learning/AI data scientists who are expecting to do machine learning/AI advanced-type things, and if the immediate needs are "Hey, take this disparate data and assemble it together and get it in front of people", there could be some job satisfaction issues that you might run into just because -- you know, I'm not saying that all people like that would think that that sort of thing is beneath them... I think a lot of people enjoy dipping into that at some point...

**Chris Benson:** Or they might...

**Daniel Whitenack:** Yeah, they might. I mean, if a year goes by, two years go by, and the only thing they've done is write SQL, there's a mismatch. At the same time, like you were saying, you could hire in the other way, and then have trouble advancing into the more sophisticated analyses, and that sort of thing. So what is your take on how to parse out that hiring bit and how to think about who you should really be bringing in? Because that could create a lot of issues.

**Chris Benson:** I have an answer for that, but I think it's a little bit of a cheat on your question... Because I think the question assumes that you don't necessarily know all the things. But I've had the benefit of a little bit of experience going through this process several times now, in several different organizations... So for me, I would hire in to reflect the entire workflow. So I know what a good data science including deep learning workflow looks like these days, from beginning to end; all of those things that have to happen, from understanding the problem, to identifying what kinds of models need to be there, to how you would implement them, what kind of equipment you need for those models, the software, how you do the DevOps or dev sec ops to get those, all the way out to deployment to production.

So from the early conception, almost at the business level, all the way through those various steps to the end, and you have something out there, it's a model that's wrapped in software that's doing something productive in the world. At this point I would catalog those -- I know how much effort roughly would go into each of those areas, a ballpark, and I hire against those levels of effort in those different stages to try to get the complete team. And depending on what the budget is and how many people, I will group some of those tasks together, and figure out what that is. It also depends on the candidates I talk to. I may make a change; if I get a particularly capable candidate, then that can change how I'm thinking about things on a tight budget.

**Daniel Whitenack:** \[19:52\] What if the CEO that hires you in is expecting some cool AI/data science/machine learning type things, but what you find out very quickly is that that actually isn't the most immediate need; the most immediate need is data aggregation and getting some metrics in front of people. How do you handle that situation with your leadership? Any thoughts there?

**Chris Benson:** First of all, they always expect that. That's not like "What if?", that is every time. Because the people responsible for that, even people who are usually supposed to be technical leadership, are beyond the details. They're making the decision, they're far enough along in their career because they may not be handling those technical details on a day-to-day basis, and therefore they don't really understand anymore, even if they think they do. And so there's a gentle education process and there's a discussion of what happens based on, you know, if you just run forward and try to do deep learning when you're not set to be able to do it effectively - you're running into a wall. And the harder you try to do it, the faster you're running into that brick wall. So there's a bit of an education process.

**Daniel Whitenack:** And also, even if there's a clear opportunity to move into some AI-related work and machine learning stuff, my experience is the right data that you need is usually very hard to get...

**Chris Benson:** Indeed.

**Daniel Whitenack:** ...or it's very fragmented. This article also talks about fragmented data, and that sort of thing. So I've been in places where you want to ask the question like "Okay, to train this model I need all of this type of data", but that's a sort of anti-pattern, because people are used to -- let's say the example is in a financial institution, or something like that. But previous maybe support people or customer service people, or even analysts or whatever, they're maybe used to looking at a very small set of transactions, or at a single transaction, or a single user and all the things that's gone on for that user... So when you ask a question "Give me all the transactions of this type", it's sort of an anti-pattern for how they've been looking at the data, and their systems aren't really set up for that sort of query even. So it may be that you have to push the infrastructure, rethink how you're getting the data, or the patterns that people are pulling data in order for you to even set up your problem, and have success in doing any type of modeling.

**Chris Benson:** I have had that same experience, and ironically, just because it's at the top of my mind, I'm thinking about that same previous employer, large company with a well-known name, and lots of physical hardware products that come out of that organization - they collect a fair amount of telemetry from those various products. But what we discovered was based on the things that we wanted deep learning models to do, and ways of improving that product's capabilities and the user experience, that most of the telemetry was absolutely useless for our purposes. It was great for figuring out what went wrong with the product after the fact, but it didn't actually -- it couldn't be used to teach a model how to more effectively do the capability. And I think that's a common -- in my experience, we've seen that across products, and I think that that would probably hold true across many organizations, where you may collect a lot of data, but that doesn't mean it's the right data, and it's not the data that's gonna help you get where you wanna go.

**Daniel Whitenack:** When you enter an organization and you're building a data team, you start interacting with product teams, and customer support, and that sort of thing... If those teams aren't yet data-driven, what are some of the things that you think motivate those teams, if it's not data? How are they making their decisions in a non-data-driven way? Because that's often what I've seen - I start interacting with a product team or something like that, and they aren't making data-driven decisions.

\[24:17\] One of the ways to think about how to change that culture is to think about what is motivating them. What has been your experience in the past in terms of the culture of the teams that you start interacting with when you build a data team?

**Chris Benson:** I think that's a huge issue, the word being "culture". Because in general, to generalize, based on at least what I've seen, when teams aren't using data to drive their decisions in an explicit, objective matter, then they're usually relying on experts, or at least self-proclaimed experts... And those decisions often are somewhat arbitrary, and oftentimes not consistent even with that person's other decisions across time and across similar situations. And in doing that, there is a belief, because they've build a business on it.

So this is one of those kind of political/cultural things that is deeply entrenched in an organization, and that you as the new leader of a data science team are forced to contend with... And it's a really hard problem. It's a hard nut to crack. They may have run years or even decades on that approach. so you have to find a way to convince them that there is a better way and that they'll get better results from that... Because in their opinion, they've gotten good results, which is why they're still doing it. If they weren't getting some level of result, it would have already passed. But it's your job, usually in the very early days, to figure out how to address those perceptions.

**Daniel Whitenack:** I'm struck by the scenario that's talked about in this article from Erik. He talks about a sort of mid-stage startup around 10 million... So that's about the size that my wife's business is, and looking at her marketing and sales/customer service department - if you think about that early stage, like you were talking about, it was basically her, she built up a ton of expertise and internal knowledge in terms of what was working and what was driving sales, and that basically boosted the company to mostly where it's at. But then you start thinking "Okay, well it's at a size where we're hiring in marketing people, or people that are supposed to be driving sales." Is it reasonable to assume that each of those people are going to have both the ownership over the business and the drive to build up that level of internal knowledge, and there's gonna be appropriate knowledge transfer between all of these people coming in? It's just not the case.

Like you say, you hit this wall where "Now how do we be creative, how do we try new things, and how do we make sure that we're driving new sales, and growing?" It has to be data-driven at that point. But the culture wasn't sort of set up that way organically. Not because they weren't wanting to be that way, but because it just sort of organically grew into this department where they're doing the things that, like you say, they know worked to some degree, and they felt like were still working... So I think now, in her company, they're doing a lot of thinking about "How do they drive that data-driven culture in marketing?" And some of it is just the very simple stuff that even Erik talked about in his article, like "Do people understand how UTM codes and website traffic works?" There needs to be some knowledge sharing there, and then there needs to be common data gathering, like "Okay, we've got this stuff over here in Facebook Pixel, and this stuff over here in Google Analytics, and this stuff over here in Shopify, and this stuff over here in these random places..." No one can really coalesce around anything if all of that is fragmented out, so there needs to be data aggregation together, there needs to be a common way to look at it...

\[28:20\] And then, you know, building that culture - it's also about people's motivation. You have to think about "If I'm gonna show something to this marketing person, how are they motivated by that?" I mean, it could be like commissions, or something. If you make this much off of Facebook ads, then you get this commission or this incentive. Well, pretty quickly they're gonna wanna know how much they're making off of Facebook ads, and if they're not setting up their UTMs right and they're not using the common systems where data is coming in, then they're not gonna be able to know

**Chris Benson:** You know, as you're telling me that and I'm listening kind of just these normal struggles of your wife's business going through this, and I'm struck with the fact that you have a brilliant wife, who's really good at what she does...

**Daniel Whitenack:** Oh, definitely.

**Chris Benson:** ...and you are really good at what you do.

**Daniel Whitenack:** Sort of. \[laughter\] I'll give her the --

**Chris Benson:** But at least she has the benefit -- she's a brilliant businessperson, and I love talking to her and I love learning from her, but she also does have the benefit of being married to you, and you're able to put these things in front of her. Most businesspeople, as smart as they are, don't have such an intimate fountain of knowledge about these particular topics. They know their business, but they don't necessarily have someone who can inform then all these data points. They can hire people to do that, but to your point, those people may not be quite as motivated as you are as a business owner, or the spouse of a business owner.

**Daniel Whitenack:** It kind of brings a lot of weight to this whole building a data team side of things, because I think about -- let's say I'm not in the picture and she hires a data person to figure out how to make our company data-driven and using modeling and all this stuff, and she hires that person... And that person spends all their time on fancy deep learning stuff, but doesn't address these basic issues of like "How does the marketing team operate? What's the culture? What numbers do they need to see in front of them?"

**Chris Benson:** It happens all the time.

**Daniel Whitenack:** That actually could -- I mean, I'm not saying it would take down the business, but it's gonna make a significant negative impact on the business.

**Chris Benson:** Absolutely.

**Daniel Whitenack:** Because it's not what's needed, right? So I think that people coming into these sorts of positions need to be -- not scared, but sober-minded in the sense of really having the perspective of what are the needs of the business, rather than what's the coolest project that I can work on.

**Chris Benson:** And what's the way I build my resume, or what's the way I get the training class that I really would love to do, but isn't necessarily directly in line with what we're trying to accomplish at the organizational level. So yeah, there's a huge risk. You're able to bring purity because all you care about is the success of the organization, but that's not always the case.

**Daniel Whitenack:** One of the other points that Erik brings up which I think is really interesting is executive support for ML/AI type things... And this sentiment that sometimes comes up when you're an AI or machine learning person, maybe you've dealt with all of the sort of getting metrics in front of people thing, there is a really important problem that you think is solved really well by machine learning and AI, you've trained a model or whatever it is, and you're genuinely convinced that this is a meaningful thing that you've done, that has great benefit for the company, but you try and try to build support for this, and you get nothing.

\[32:09\] What is in your experience maybe going on in that situation, where you have the solution but you're having trouble either helping people in the organization understand it, or understand the value, or understand the benefit, and buy into it and support it?

**Chris Benson:** It's your job to communicate that as the leader of the effort, or as the visionary who understands what's possible. You have to be able to explain it, but without diving into all the technical details, and you have to be able to -- maybe you're either not using data science, or maybe you're using more traditional mechanisms in data science, and you know that a convolutional neural network or a natural language processing model has a particular strength in a certain area - you've gotta find a way to communicate that, and doing that by, to some degree, dumbing it down; and I don't mean that in a derogatory way, I mean that - your audience is not as technical as you are, so you have to get that communication at the level that they get the value, but you have to abstract it to a point where they're gonna understand that. But that's really on you, it's not on them. It's your job to show them "If I look at other similar fields, and they used to do it this way, and here's a paper or an article about that... And this other company, who has something similar to ours, or maybe has a similar interest in this particular task, did something and they've gone all-in after testing it - this is why. These are the basics of this, that's the basic of that. There's a definite advantage, we should invest in that."

And also know when not to invest... Because we've come through an age the last few years where so many people are just wanting to do AI so they can say they're doing AI, and there's a lot of things that deep learning is not the best thing for, or at least it's way more expensive than other options that are equally as good or nearly as good. So it's finding somebody who understands that to lead your effort, and who can communicate that effectively to all the stakeholders.

**Daniel Whitenack:** I think that's why maybe certain tools that have come out recently, like let's say a Streamlit or something like that, can actually be incredibly powerful to solve this problem... Because would you wanna develop your whole product in Streamlit? Maybe, maybe not, depending on what it is... But could you use that tool in order to prototype something out and demonstrably show the value of what you're doing? Yeah, definitely.

Prototypes and minimal viable things are really valuable in this case, and often -- the way that Erik puts this is sometimes the data team just doesn't take it upon themselves to get the work into a place where it demonstrates value, and is reasonably easy to ship. So you could have a Jupyter Notebook, or you could run a model on your GPU server, and then do your tests on your GPU server and show that you get a 90% accuracy or whatever it is, and then your executive team is like "Oh, that's great. Let's ship it. How do we get it into this product?" and if your answer is "Oh, I haven't thought about that yet" or "It only runs on the GPU server" or "I don't know how to extract it. We have to figure out that problem. Do you wanna invest in that?", it doesn't really expire a lot of confidence.

**Chris Benson:** It doesn't.

**Daniel Whitenack:** \[35:43\] So I think just that one step of "Do we expect data teams or data scientists to actually build robust products?" Maybe in certain cases they are part of that, like in smaller startups and that sort of thing... In larger companies maybe not. But should they be expected to maybe go that little extra mile to create a prototype that demonstrates value and gets things in front of people in a meaningful way, even if it doesn't scale all the way up? I think that's a huge, huge point that I don't see emphasized that much... You know, because you see a lot emphasized about getting your models training well, and evaluating well, but not this sort of prototyping of the problem.

**Chris Benson:** I agree with that completely. I think that people over-talk it a little bit early on and don't recognize the value of lightweight prototyping to help you figure your way through it, figure out what it is that you need and prove out that what you're thinking is actually accurate... Because if you think of how many organizations build things that actually are not very useful in the end, or don't have the audience that they originally expected, you can solve that in part with prototyping, and help yourself hone in on it.

And I think that a lot of organizations interpret that as scary in away, or potentially as long as expensive, so they make the mistake of trying to talk their way through it... And I've seen that through my entire career. In my earliest days there was no such thing as Agile. That didn't happen for a while actually, until eventually this Agile movement around the beginning of the 2000's came about... And it's taken the next 20 years for that mindset to really take hold in a broad sense. So you need to try stuff out, and you need to be ready to go off and do a coding spike on something and figure out with a simple model whether this is doable. Or if it's doable, can you deploy it? Is it deployable in a reasonable way? Do you have resources where you need to deploy it?

So your points are well made. These are all things that you need to be thinking about when you're building these teams and you're looking for the people with the right mindset, and the right skillsets, so that you can be successful.

**Daniel Whitenack:** And on your data teams that you've built in the past, in terms of the communication between the data team and maybe organizational units that that data team is serving - let's say marketing, or supply chain, or whatever it is - how does the communication work between those external, or internal, but other organizational units, and the data team work? In your experience often does that flow through the management of the data team, or does that flow directly through the individual data scientist working on various products?

**Chris Benson:** So I don't think there's any standardized way, and I've seen that happen in all sorts of different ways, and some of them are formal and some of them are informal, just because people are talking... But I've rarely seen great integration in that capacity. I've rarely seen that kind of inter-functional communication, and the translation required for them to understand each other to happen well and seamlessly and consistently. But it definitely helps... And this goes back to that thing we talked about the very beginning, it's culture. If you don't evolve your growing organization with the right culture to take advantage of this, it's really hard to make that change down the road. And you may have to, but you're really gonna have to consciously set some things aside that maybe were long-time valued processes.

**Daniel Whitenack:** And I think this is like a growth area for me that I have seen recently - oftentimes when you're building a team, if you're the one building the team, it's most natural for all communication and project communication and all of that to sort of flow through you as the management of the data team... And at a certain point -- so I think that's probably good in the beginning, because you're setting some standards, you're setting workflows, all of this stuff, and you sort of need to guard that a little bit, I think; that's probably reasonable. But as the team grows and as the number of projects grows, at some point you become the bottleneck, right?

**Chris Benson:** \[40:11\] You do.

**Daniel Whitenack:** If all the communication from all of these different projects in the different organizational units are flowing through you, then pretty quickly the queue builds up and stuff is falling through the cracks. So I feel that right now in terms of the teams that I'm helping build, there needs to be this transition to more embedded communication, where the people working on different projects are feeling the freedom to have those communication while still trying to maintain standards and all of that, of course, and make sure that projects are done well. There still needs to be data management, but I think things need to get more decentralized over time.

**Chris Benson:** It does. And that data management that you're talking about has to become human management. It has to be recognizing your individuals for what they are, and what their capabilities are, and understanding that they're all different, and developing a good understanding of what each of those individuals can and can't do well, on a longer spectrum, and then being able to give them those responsibilities where they both can be successful, but also have room to grow... And that is really hard to do; it's super-easy for me to say that, and it's super-hard to execute that. That's what you have to do if you're gonna grow past those early stages of putting it all together.

**Daniel Whitenack:** I was really inspired by this article, just in terms of understanding where I've been at in the past and how I've been able to grow into certain data teams, but also I think it's a really great way to frame some of this up in a creative way... So thank you, Erik, for writing this and putting in the time to do it. You're welcome on the show anytime to talk through other things along with this... So if you're out there, feel free to join us sometime.

In these Fully Connected episodes we usually also try to give a couple learning resources related to the topic that we're talking about... This week I wanted to mention a book from one of our past guests, Mike Bugembe, "Cracking the Data Code", which is a great book where he talks a lot about data culture and creating a data-driven culture in your business. And then also, there's a book by also one of our previous guests, Hilary Mason and DJ Patil called "Data Drive: Creating a Data Culture." It's more like a booklet; I think you can get it in some cases for free, like on your Kindle or something like that... It's a good little read.

**Chris Benson:** Those are great choices right there.

**Daniel Whitenack:** Yeah, there's definitely resources out there, and although you and I have had experiences, I definitely respect the opinions of people like Erik and Hilary and DJ and Mike Bugembe and others, who really have been able to scale things like that up... Because like you said, the human problem is maybe the main thing that you're dealing with as you're building a data team; it's not so much your ability as a data team to do things, but how you relate to other teams in your organization, how you can be gracious and clear and tenacious and creative and all those things together, and not burn a bunch of bridges and end up in a bad situation.

**Chris Benson:** Yeah, that's really important. And you really have to respect all of those people, including the diversity of their differences... Because they did not have your experiences, they did not grow up thinking that data was the thing they were gonna spend their time on... You have to position the value in a way that they understand and that they can also value... And it's crossing that chasm that is the key to success there.

**Daniel Whitenack:** Well, thanks for having this chat. I enjoyed hearing your stories, Chris.

**Chris Benson:** I enjoyed hearing yours as well. We have good ones.

**Daniel Whitenack:** Alright, I'll see you next week.

**Chris Benson:** Okay, take care.