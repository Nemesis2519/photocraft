// Photoshop's own Select Subject on every photo in a folder, saved as a mask PNG in another:
// white where selected, black where not, grey in proportion to a soft selection. The mask is
// made by filling the selection with white over black in RGB, so a pixel selected a fraction
// a comes out as a*255 (a grayscale conversion would bend the values through a dot-gain curve).
app.displayDialogs = DialogModes.NO;
// Folders come as script arguments: DoJavaScriptFile(path, [photos, masks]).
if (typeof arguments == "undefined" || arguments.length < 2) throw new Error("pass [photos folder, masks folder]");
var src = Folder(arguments[0]);
var out = Folder(arguments[1]);
if (!out.exists) out.create();
var files = src.getFiles(/\.(jpe?g|png)$/i);
var log = [];

function color(v) {
    var c = new SolidColor();
    c.rgb.red = v; c.rgb.green = v; c.rgb.blue = v;
    return c;
}

for (var i = 0; i < files.length; i++) {
    var f = files[i];
    var stem = decodeURI(f.name).replace(/\.[^.]+$/, "");
    var doc = null;
    try {
        doc = app.open(f);
        if (doc.mode != DocumentMode.RGB) doc.changeMode(ChangeMode.RGB);
        if (doc.bitsPerChannel != BitsPerChannelType.EIGHT) doc.bitsPerChannel = BitsPerChannelType.EIGHT;
        var start = new Date().getTime();
        var d = new ActionDescriptor();
        d.putBoolean(stringIDToTypeID("sampleAllLayers"), false);
        executeAction(stringIDToTypeID("autoCutout"), d, DialogModes.NO);
        var took = (new Date().getTime() - start) / 1000;
        var subject = doc.artLayers.add();
        doc.selection.fill(color(255), ColorBlendMode.NORMAL, 100, false);
        doc.selection.deselect();
        var ground = doc.artLayers.add();
        doc.selection.selectAll();
        doc.selection.fill(color(0), ColorBlendMode.NORMAL, 100, false);
        doc.selection.deselect();
        ground.move(subject, ElementPlacement.PLACEAFTER);
        doc.flatten();
        doc.saveAs(File(out.fsName + "/" + stem + ".png"), new PNGSaveOptions(), true, Extension.LOWERCASE);
        log.push(stem + ": saved, Select Subject took " + took + "s");
    } catch (e) {
        log.push(stem + ": FAILED " + e);
    }
    if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
}
log.join("\n");
