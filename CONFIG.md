### SadedeGel Configuration

In time we keep adding different configurable parameters into sadedeGel 
to support all popular variations per NLP task/subtask. Such as

* Tokenizer (word tokenizer)
* Term Frequency (tf) strategy
* Inverse Document Frequency (idf) strategy
  * Or parameters
* Embedding types

#### Default Parameters
Starting with version `0.16` default values of those parameters are shipped with library in `default.ini` file

```ini
[default]
tokenizer = bert

[tf]
method = binary

# default value for Double Log method
double_norm_k = 0.5

[idf]
method = smooth
```

#### Overwriting defaults
Users may prefer to overwrite those parameters by creating `~/.sadedegel/user.ini` file.
You may completely/partially include parameters available in `default.ini` file.

#### Get all Parameters

We have introduced a commandline to show current values to be used by sadedegel (unless user overwrites them at runtime)

```shell script
sadedegel config
```

```
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ section ┃ parameter_name ┃ current_value ┃ default_value ┃ description                              ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ default │ tokenizer      │ bert          │ bert          │ Word tokenizer to use                    │
│ tf      │ method         │ raw           │ binary        │ Method used in term frequency            │
│         │                │               │               │ calculation                              │
│ tf      │ double_norm_k  │ 0.5           │ 0.5           │ Smooth parameter used by double norm     │
│         │                │               │               │ term frequency method.                   │
│ idf     │ method         │ smooth        │ smooth        │ Method used in Inverse Document          │
│         │                │               │               │ Frequency calculation                    │
└─────────┴────────────────┴───────────────┴───────────────┴──────────────────────────────────────────┘
```

If you get a misbehaviour compared to examples we provide, 
always ensure that your local settings are not overwritten by your user level configuration.