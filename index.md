---
layout: default
title: About Artificial Intelligence
description: Different Kinds of Artifical Intelligence
---

# About Artificial Intelligence

This Github website currently contains two part.

## Neural Network Model Study
Have studied different kinds of Neural Network Models

<ul>
  {% for doc in site.ToNN %}
    <li>
      <a href="{{ doc.url | relative_url }}">{{ doc.title }}</a>
    </li>
  {% endfor %}
</ul>

## Face Anti-Spoofing

<ul>
  {% for doc in site.FAS %}
    <li>
      <a href="{{ doc.url | relative_url }}">{{ doc.title }}</a>
    </li>
  {% endfor %}
</ul>
